# Versioning Policy

The project version is release metadata controlled by the repository owner.
It must not be changed merely because a feature, fix, profiling hook,
performance milestone, Docker integration, or dependency pin is added.

Development commits keep the current project version. Consumers that need a
specific development state must pin the full Git commit instead of inventing a
new package version.

A version change requires an explicit owner decision that names the target
version. Do not infer release authorization from a branch name, milestone,
upstream sync, or the scope of a code change.
