# Contributing

Thanks for taking a look at this project! It started as a personal learning project (part of my data science portfolio), but I'm happy to take suggestions, bug reports, or improvements.

## How to contribute

1. **Fork** the repository
2. **Create a branch** for your change
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Notebooks: please clear cell outputs before committing if the outputs are large (`Kernel > Restart & Clear Output`), so diffs stay readable.
   - Web app: keep it to `app.py` + `templates/` + `static/` — no extra framework layers, this is meant to stay a simple demo app.
4. **Test locally**
   - Make sure the notebooks run top-to-bottom without errors.
   - Make sure `python app.py` still boots and the form submits correctly.
5. **Commit** with a clear message
   ```bash
   git commit -m "Add: short description of what changed"
   ```
6. **Push and open a Pull Request** against `main`, describing what you changed and why.

## Reporting issues

If you find a bug or something confusing, please open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce (Python version, OS, whether you have a GPU, etc.)

## Ideas I'd welcome help with

- Support for SQuAD 2.0 (unanswerable questions)
- A confidence-score display in the web app
- Dockerfile for easier deployment
- Unit tests around the Flask routes

## Code style

- Keep notebook cells small and readable — one logical step per cell.
- Keep the Flask app minimal — this project intentionally avoids a `src/` package structure, so please don't introduce one in a PR; new logic should live in `app.py` or in a notebook.

Please be respectful and constructive in issues/PRs — this is a learning project and I'm still improving as I go!
