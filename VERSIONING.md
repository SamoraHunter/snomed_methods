# Versioning Strategy

This project follows [Semantic Versioning (SemVer)](https://semver.org/) with the version format `MAJOR.MINOR.PATCH`.

## Version Format

```
MAJOR.MINOR.PATCH
```

- **MAJOR**: Breaking changes (API breaking, major architecture overhaul)
- **MINOR**: New features (backwards compatible additions)
- **PATCH**: Bug fixes (backwards compatible bug fixes)

## Version Numbers

- Current version: **v0.1.0**
- Initial stable release: Will be v1.0.0 upon reaching production readiness
- Development versions: Use `-dev` suffix (e.g., `v0.2.0-dev`)

## Versioning Guidelines

### When to increment MAJOR

- API breaking changes (function signatures, return types, class interfaces)
- Major architectural changes (new module structure, renamed core classes)
- Removing or renaming public functions/classes
- incompatible configuration format changes

**Example**: v1.2.3 → v2.0.0

### When to increment MINOR

- New features added to existing modules
- New modules added (backwards compatible)
- New parameters added with default values
- New configuration options

**Example**: v1.2.3 → v1.3.0

### When to increment PATCH

- Bug fixes that don't change API behavior
- Performance improvements
- Documentation updates
- Test additions/changes

**Example**: v1.2.3 → v1.2.4

## Release Process

1. **Development**: Work on feature branches
2. **Pre-release**: Tag with `-dev` suffix for testing (e.g., `v0.2.0-dev`)
3. **Release Candidate**: Tag as `vX.Y.Z-rc.N` for RC releases
4. **Stable Release**: Tag as `vX.Y.Z` and create GitHub release

## Version Tracking

- Git tags: All releases are tagged with `vX.Y.Z`
- CHANGELOG.md: Each version includes changelog entries
- pyproject.toml: Version field updated on each release

## Compatibility Policy

- **Patch versions**: Always backwards compatible within same MINOR series
- **Minor versions**: Backwards compatible within same MAJOR series
- **Major versions**: May have breaking changes (update required)

## Notes for 0.x Versions

Before v1.0.0, the API is unstable:
- Minor versions may introduce breaking changes
- Features may change or be removed without warning
- Once v1.0.0 is reached, semver guarantees apply strictly
