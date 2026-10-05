# PGO training set

Ten public scenario configs, copied verbatim from `regression_suite/scenarios/`. The Docker build
runs an instrumented `fastsim-native` over them so that gcc can lay out the event loop's branches
from a real run (`-fprofile-use`). The profile only changes code layout and inlining decisions:
the floating-point flags are the same (`-ffp-contract=off -fno-fast-math`), so outputs are
identical with or without it. If the training run fails (for example under an emulator), the build
continues without a profile.
