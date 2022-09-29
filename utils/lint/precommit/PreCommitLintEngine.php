<?php
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/**
 * Engine which works around `arc lint`s design to run file by file
 *
 * With vanilla `arc lint`, `pre-commit` will be run in the following way
 *
 * ```sh
 * pre-commit run --file file1
 * pre-commit run --file file2
 * ```
 *
 * This is inherently slow as it doesn't utilize `pre-commit`s parallelization
 * Instead, `pre-commit` should be run in the following way
 *
 * ```sh
 * pre-commit run --file file1 file2
 * ```
 *
 * in order to get a significant speed-up
 *
 * This engine utilizes a work around where only one path is added for linting.
 * In this way, the `getPathArgumentForLinterFuture` in the `PreCommitLinter`
 * is run only once, and we can modify the output to run on all files.
 */
final class PreCommitLintEngine extends ArcanistLintEngine {
  public function buildLinters() {
    // Create the PreCommitLinter
    $pre_commit_linter = new PreCommitLinter();

    // Set the static path of pre_commit_linter
    $pre_commit_linter::$_paths = $this->getPaths();

    // The linter needs one file in order to run
    // We here set it to the first path of getPaths
    if (!empty($this->getPaths())) {
      $pre_commit_linter->addPath($this->getPaths()[0]);
    }

    return [$pre_commit_linter];
  }
}
