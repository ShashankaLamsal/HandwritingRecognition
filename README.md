For Handwritten Word Recognition project using Python with Flask.


# Git Workflow with Main, Staging, and Dev Branches


1. **dev** - Development commits.
2. **staging** - Stable versions.
3. **main** - Final releases.

## Setting Up the Repository

1. **Initialize Git:**
   ```bash
   git init
   ```

2. **Create branches:**
   ```bash
   git branch -M main
   git checkout -b staging
   git checkout -b dev
   ```

3. **Push to GitHub:**
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main staging dev
   ```

## Workflow

1. **Commit to `dev`: [DEFAULT]**
   ```bash
   git checkout dev
   git add .
   git commit -m "Your message"
   git push -u origin dev
   ```

2. **Merge to `staging`:**
   ```bash
   git checkout staging
   git merge dev
   git push origin staging
   ```

3. **Merge to `main`:**
   ```bash
   git checkout main
   git merge staging
   git push origin main
   ```

## Notes
- Test changes in `dev` before merging to `staging`.
- Ensure `staging` is stable before merging into `main`.
- Write clear commit messages.

