"""Public Kitchen native decode/deferred integration with Triton disabled."""
import importlib
import pytest
import torch
import comfy_kitchen as ck
from comfy_kitchen.registry import registry

delta=importlib.import_module("comfy_kitchen.gated_delta")
cache_api=importlib.import_module("comfy_kitchen.int8_decode")


@pytest.fixture
def native_xpu():
    if not torch.xpu.is_available():pytest.skip("requires XPU")
    from omni_xpu_kernel import kitchen
    if not kitchen.supports_int8_decode() or not kitchen.supports_gated_delta_deferred():
        pytest.skip("requires native LLM companion")
    was_disabled="triton" in registry._disabled
    registry.disable("triton")
    try:yield kitchen
    finally:
        if not was_disabled:registry.enable("triton")


def test_native_decode_public_route(native_xpu):
    assert ck.flash_attention_decode_gqa_is_available(torch.device("xpu:0"))
    q=torch.randn(1,4,4,256,device="xpu",dtype=torch.bfloat16)
    k=torch.randn(1,2,513,256,device="xpu",dtype=q.dtype);v=torch.randn_like(k)
    length=torch.tensor([513],device="xpu",dtype=torch.int32)
    ref=importlib.import_module("comfy_kitchen.flash_attention")._decode_gqa_torch(q,k,v,length)
    torch.testing.assert_close(ck.flash_attention_decode_gqa(q,k,v,length),ref,rtol=.01,atol=.002)
    legacy=ck.flash_attention_decode(q[:,:,:1].transpose(1,2),k.transpose(1,2),v.transpose(1,2),length)
    expected=native_xpu.decode_attention(q[:,:,:1],k,v,length,False)[0]
    torch.testing.assert_close(legacy,expected.reshape_as(legacy),rtol=0,atol=0)


def test_native_cache_public_route(native_xpu):
    assert cache_api.is_available(torch.device("xpu:0"))
    cache=cache_api.Int8DecodeCache(1,1,1024,"xpu:0")
    k=torch.randn(1,1,1024,256,device="xpu",dtype=torch.bfloat16);v=torch.randn_like(k)
    length=torch.tensor([513],device="xpu",dtype=torch.int64)
    cache.update(k,v,length,True)
    q=torch.randn(1,2,4,256,device="xpu",dtype=k.dtype)
    out,lse=cache.attend(q,length)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    length.zero_();cache.update(k,v,length)
    out,lse=cache.attend(q,length)
    assert torch.equal(out,torch.zeros_like(out)) and torch.isneginf(lse).all()


def test_public_gqa_compile_retains_native_dispatch(native_xpu):
    q=torch.randn(1,4,2,256,device="xpu",dtype=torch.bfloat16)
    k=torch.randn(1,2,257,256,device="xpu",dtype=q.dtype);v=torch.randn_like(k)
    length=torch.tensor([257],device="xpu",dtype=torch.int32)
    graphs=[]
    def backend(gm,inputs):
        graphs.append(gm.code)
        return gm.forward
    compiled=torch.compile(ck.flash_attention_decode_gqa,backend=backend,fullgraph=True)
    expected=ck.flash_attention_decode_gqa(q,k,v,length)
    torch.testing.assert_close(compiled(q,k,v,length),expected,rtol=0,atol=0)
    assert any("omni_xpu.kitchen_decode_attention" in g for g in graphs)
    with ck.use_backend("eager"):
        eager=ck.flash_attention_decode_gqa(q,k,v,length)
        torch.testing.assert_close(compiled(q,k,v,length),eager,rtol=0,atol=0)
    assert "omni_xpu.kitchen_decode_attention" not in graphs[-1]


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize("steps",[1,4,8])
def test_native_deferred_matches_upstream_torch_replay(native_xpu,dtype,steps):
    torch.manual_seed(173)
    b,h,hk,d,hd=2,6,2,128,64
    channels=2*hk*d+h*d
    wa=torch.randn(h,hd,device="xpu",dtype=dtype)*.05;wb=torch.randn_like(wa)*.05
    dt=torch.randn(h,device="xpu");gd=-torch.rand(h,device="xpu")-.5
    nw=torch.rand(d,device="xpu",dtype=dtype)+.5
    cw=torch.randn(channels,4,device="xpu",dtype=dtype)*.1
    cb=torch.randn(channels,device="xpu",dtype=dtype)*.1
    state=torch.randn(b,h,d,d,device="xpu")*.1;ref_state=state.clone()
    conv=torch.randn(b,channels,3,device="xpu",dtype=dtype)*.1;ref_conv=conv.clone()
    qkv,proj,gates,sums=delta.deferred_buffers(b,channels,h,hk,dtype,torch.device("xpu"))
    for buf in (qkv,proj,gates,sums):buf.zero_()
    rq,rp,rg,rs=(t.clone() for t in (qkv,proj,gates,sums))
    ctl=torch.tensor([0,0],device="xpu",dtype=torch.int32)
    assert delta.deferred_is_available(torch.device("xpu:0"))
    for parity,pending in [(0,0),(1,steps),(0,max(0,steps-1)),(1,0)]:
        ctl.copy_(torch.tensor([pending,parity],device="xpu",dtype=torch.int32))
        x=torch.randn(b,steps,hd,device="xpu",dtype=dtype)
        projections=torch.randn(b,steps,channels,device="xpu",dtype=dtype)*.2
        z=torch.randn(b,steps,h*d,device="xpu",dtype=dtype)
        delta._conv_deferred_torch(projections,ref_conv,cw,cb,rp,rq,ctl)
        expected=delta._decode_deferred_torch(x,wa,wb,dt,gd,ref_state,hk*d,hk,d**-.5,
            z,nw,1e-6,rq,rg,rs,ctl)
        delta.deltanet_conv_step_deferred(projections,conv,cw,cb,proj,qkv,ctl)
        actual=delta.gated_delta_decode_deferred(x,wa,wb,dt,gd,state,hk*d,hk,d**-.5,
            z,nw,1e-6,qkv,gates,sums,ctl)
        tol=1e-5 if dtype==torch.float32 else .01
        torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
        torch.testing.assert_close(state,ref_state,rtol=tol,atol=tol*.1)
        torch.testing.assert_close(conv,ref_conv,rtol=0,atol=0)
        torch.testing.assert_close(proj,rp,rtol=0,atol=0)
        torch.testing.assert_close(qkv,rq,rtol=tol,atol=tol)
        # Matching state after pending=0 also proves the current chain remains uncommitted.


def test_decode_admission_respects_native_capability_and_override(monkeypatch):
    from comfy_kitchen.backends.xpu import decode
    from omni_xpu_kernel import kitchen
    monkeypatch.setattr(torch.xpu,"is_available",lambda:True)
    monkeypatch.setattr(torch.xpu,"device_count",lambda:1)
    monkeypatch.setattr(registry,"is_available",lambda name:name in ("xpu","eager"))
    monkeypatch.setattr(kitchen,"supports_decode_attention",lambda:True)
    assert decode.is_available("xpu:0")
    assert not decode.is_available("xpu:1")
    with ck.use_backend("eager"):assert not decode.is_available("xpu:0")
    monkeypatch.setattr(kitchen,"supports_decode_attention",lambda:False)
    assert not decode.is_available("xpu:0")


@pytest.mark.parametrize("magnitude",[0.,1e-8])
def test_deferred_zero_and_small_qk_normalization(native_xpu,magnitude):
    x=torch.zeros(1,2,8,device="xpu",dtype=torch.bfloat16)
    w=torch.zeros(2,8,device="xpu",dtype=x.dtype)
    dt=torch.zeros(2,device="xpu");gd=-torch.ones_like(dt)
    state=torch.randn(1,2,128,128,device="xpu")*.1;before=state.clone();ref=state.clone()
    qkv,proj,gates,sums=delta.deferred_buffers(1,512,2,1,x.dtype,x.device)
    qkv.fill_(magnitude);gates.zero_();sums.zero_()
    ctl=torch.tensor([0,0],device="xpu",dtype=torch.int32)
    z=torch.ones(1,2,256,device="xpu",dtype=x.dtype);nw=torch.ones(128,device="xpu",dtype=x.dtype)
    expected=delta._decode_deferred_torch(x,w,w,dt,gd,ref,128,1,128**-.5,z,nw,1e-6,
        qkv,gates.clone(),sums.clone(),ctl)
    actual=delta.gated_delta_decode_deferred(x,w,w,dt,gd,state,128,1,128**-.5,z,nw,1e-6,qkv,gates,sums,ctl)
    torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
    assert torch.equal(state,before)


def test_deferred_graph_replays_device_parity_and_accepted_prefix(native_xpu):
    b,s,h,hk,d,hd=1,4,2,1,128,32;c=512
    dtype=torch.bfloat16
    x=torch.randn(b,s,hd,device="xpu",dtype=dtype)
    projections=torch.randn(b,s,c,device="xpu",dtype=dtype)*.1
    w=torch.randn(h,hd,device="xpu",dtype=dtype)*.05
    dt=torch.zeros(h,device="xpu");gd=-torch.ones_like(dt)
    z=torch.randn(b,s,h*d,device="xpu",dtype=dtype);nw=torch.ones(d,device="xpu",dtype=dtype)
    cw=torch.randn(c,4,device="xpu",dtype=dtype)*.1
    state=torch.randn(b,h,d,d,device="xpu")*.01
    conv=torch.zeros(b,c,3,device="xpu",dtype=dtype)
    qkv,proj,gates,sums=delta.deferred_buffers(b,c,h,hk,dtype,x.device)
    for a in (qkv,proj,gates,sums):a.zero_()
    ctl=torch.tensor([0,0],device="xpu",dtype=torch.int32)
    def call():
        delta.deltanet_conv_step_deferred(projections,conv,cw,None,proj,qkv,ctl)
        return delta.gated_delta_decode_deferred(x,w,w,dt,gd,state,d,hk,d**-.5,z,nw,1e-6,qkv,gates,sums,ctl)
    call();call();torch.xpu.synchronize()
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):out=call()
    rc,rs,rq,rp,rg,rss=(a.clone() for a in (conv,state,qkv,proj,gates,sums))
    for parity,pending in [(1,0),(0,4),(1,2),(0,0)]:
        ctl.copy_(torch.tensor([pending,parity],device="xpu",dtype=torch.int32))
        delta._conv_deferred_torch(projections,rc,cw,None,rp,rq,ctl)
        expected=delta._decode_deferred_torch(x,w,w,dt,gd,rs,d,hk,d**-.5,z,nw,1e-6,rq,rg,rss,ctl)
        graph.replay();torch.xpu.synchronize()
        torch.testing.assert_close(out,expected,rtol=.01,atol=.002)
        torch.testing.assert_close(state,rs,rtol=.01,atol=.001)
        torch.testing.assert_close(conv,rc,rtol=0,atol=0)
