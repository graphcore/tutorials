<?php
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

final class PreCommitLinter extends ArcanistExternalLinter {

  // Variable that will filled with all paths on creation of this class
  public static $_paths = [];

  private const LINTER_NAME = "pre-commit";
  private const DEFAULT_BINARY = "pre-commit";
  private const RUN_LINTER_COMMAND = "run --file %s";
  private const INSTALL_LINTER = "Run `pip3 install pre-commit && pre-commit install && pre-commit run` and follow the install instruction";
  private const REGEX_LINTER_FAILED = '/(?<linter>[^\.]+)\.+Failed/';

  /** Return the name of linter visible to arc */
  public function getLinterName() {
    return self::LINTER_NAME;
  }

  /** Return the string representing the binary to run */
  public function getDefaultBinary() {
    return self::DEFAULT_BINARY;
  }

  /**
   * Get the argument to run for a specific file.
   *
   * NOTE:
   * Due to the work-around in PreCommitLintEngine, this will only run once,
   * see PreCommitLintEngine.php for details
   *
   * @param path The path to run the linter on (not used due to work-around)
   */
  protected function getPathArgumentForLinterFuture($path) {
    return vsprintf(self::RUN_LINTER_COMMAND, [implode(' ', static::$_paths)]);
  }

  /** Return what to print out in case the binary can not be found. */
  public function getInstallInstructions() {
    return pht(self::INSTALL_LINTER);
  }

  /** Return messages to be printed in case the linter fails. */
  protected function parseLinterOutput($path, $err, $stdout, $stderr) {
    $messages = [];

    preg_match_all(self::REGEX_LINTER_FAILED, $stdout, $matches, PREG_SET_ORDER, 0);

    if (!empty($matches))
    {
      $message = new ArcanistLintMessage();
      $message->setPath($path);
      $message->setName('pre-commit');
      $message->setSeverity(ArcanistLintSeverity::SEVERITY_ERROR);
      $message->setDescription($stdout . "\nNOTE: Tutorials uses 'pre-commit' "
        . "to run linters. You are seeing this message because pre-commit has "
        . "returned an error. You can run pre-commit directly by calling:\n"
        . "\n"
        . " > pre-commit run\n");

      $messages[] = $message;
    }

    return $messages;
  }
}
