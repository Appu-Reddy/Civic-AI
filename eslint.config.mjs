import js from "@eslint/js";
import globals from "globals";

export default [
  js.configs.recommended,
  {
    files: ["**/*.js"],
    ignores: ["node_modules/**", "dist/**", "coverage/**"],

    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module", // Change to "commonjs" if you're not using ES modules
      globals: {
        ...globals.node,
      },
    },

    rules: {
      // Best practices
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-console": "off",
      "no-var": "error",
      "prefer-const": "error",

      // Style
      "semi": ["error", "always"],
      "quotes": ["error", "single"],
      "eqeqeq": ["error", "always"],
      "curly": ["error", "all"],
    },
  },
];