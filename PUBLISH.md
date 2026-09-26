# How to publish CrossWatch for the judges

The judges need **two links**: a public GitHub repository and a public website with the live demo.
A zip file is not accepted. This guide takes about 30-45 minutes, most of it waiting for uploads and builds.
All commands are for Windows PowerShell, run inside the unzipped project folder.

## 0. Before you start

- Unzip the project somewhere outside OneDrive/Dropbox, e.g. `C:\crosswatch`.
- You need free accounts on **github.com** and **huggingface.co**.
- Install the tools (skip what you already have):

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Git.Git -e
winget install --id GitHub.cli -e
```

Close and reopen PowerShell afterwards so the new commands are found.

## 1. Fill in the team

Open `website/static/data/team.json` in any text editor and replace the three placeholder members.
For each person: `name`, `role`, `did` (what they actually did), `projects` (1-2 they are proud of) and `links`
(GitHub, LinkedIn, Portfolio; leave a link as `""` if the person does not have it and it will be hidden).
Do not add email addresses; the site is public.

## 2. Put the code on GitHub

```powershell
cd C:\crosswatch
gh auth login                      # choose GitHub.com, HTTPS, "Login with a web browser"
git init -b main
git config user.name  "Your Name"
git config user.email "<id>+<username>@users.noreply.github.com"   # GitHub > Settings > Emails shows this private address
git add -A
git commit -m "CrossWatch: traffic event detection, WIUT Hackathon 2026 CV track"
gh repo create crosswatch --public --source . --push
```

The repository is now at `https://github.com/<your-username>/crosswatch`. The upload is ~150 MB (model weights and
the annotated sample videos), so give it a few minutes.

## 3. Publish the website and live demo (Hugging Face Space)

1. On huggingface.co: Settings > Access Tokens > **Create new token**, type **Write**. Copy it.
2. Then:

```powershell
py -m pip install huggingface_hub
hf auth login                      # paste the token
py deploy\huggingface\publish.py --space <hf-username>/crosswatch --repo-url https://github.com/<your-username>/crosswatch
```

The script fills the site's Links section and the README team table, then uploads the site.
Open `https://huggingface.co/spaces/<hf-username>/crosswatch`. The first build takes ~10 minutes (status "Building").
When it shows the page, test the demo: download the 20-second sample from the Live demo section and upload it.
It should find a red-light run at 0:04 after about 2 minutes.

## 4. Save the updated links on GitHub and tag the submission

```powershell
git add -A
git commit -m "Add team and links"
git push
git tag v1.0
git push origin v1.0
```

The judges run the **tagged commit**. If you change anything later, make a new tag (v1.1) before the deadline.

## 5. Keep the website awake

Free Spaces sleep after ~48 hours without visitors. The repository has a small GitHub Action that visits the site
every 6 hours:

1. GitHub repo > Settings > Secrets and variables > Actions > **Variables** tab > New repository variable:
   name `SPACE_URL`, value `https://<hf-username>-crosswatch.hf.space` (printed by the publish script).
2. Actions tab > **keep-space-awake** > "Run workflow" once to check it works (green tick).

Also open the site yourself the day before judging.

## 6. Submit

- Repository: `https://github.com/<your-username>/crosswatch/tree/v1.0`
- Website: `https://huggingface.co/spaces/<hf-username>/crosswatch`

## If something goes wrong

- `gh`, `git` or `hf` "not recognized": reopen PowerShell after installing.
- Space shows "Runtime error": open the Logs tab on the Space page. Most often it is still building, so wait.
- Space shows "Sleeping": open it once; it wakes in 1-2 minutes. Check that step 5 is set up.
