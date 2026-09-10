# Lab 1 - Answers

## Question 1: Observe the files created, what do you think they contain? (uv init files)

Running `uv init` scaffolds a minimal Python project. It created:

- **`pyproject.toml`** - the project's manifest: name, Python version requirement, and (as we add packages) the dependency list. This is what `uv add`/`uv run` read to know what to install and which interpreter to use.
- **`.python-version`** - pins the exact Python version uv should use for this project, so the environment is reproducible across machines.
- **`README.md`** - a placeholder project description.
- **A `.gitignore`** - ignoring things like the virtual environment folder so they don't get committed.

None of these contain actual code logic yet — they're just the skeleton that lets `uv run` and `uv add` work consistently.

## Question 2: What are the created files. What do you think they are used for? And which ones should be pushed to git? (files created by `dvc init`)

`dvc init` creates a `.dvc/` folder containing:

- **`.dvc/config`** — the remote/settings configuration (non secret). This should be pushed to git so anyone cloning the repo knows which remote to pull data from.
- **`.dvc/config.local`** — where credentials/secrets get written when configured (see Question 3). This should not be pushed to git.
- **`.dvc/cache/`** — the local content-addressable store where DVC keeps the actual data blobs, keyed by hash. This is large and machine-specific, so it should not be pushed.
- **`.dvc/tmp/`** — temporary/lock files DVC uses internally, not meant to be versioned.
- **`.dvcignore`** — analogous to `.gitignore` but for DVC, listing patterns DVC should skip when scanning for files.

Only `.dvc/config` and `.dvcignore` should be committed to git — everything else is either machine-local, secret, or regenerable.

## Question 3: Where are the credentials stored? and what are the options other than --global? Should the credentials be pushed to github?

Running `dvc remote modify origin --global user/password ...` writes the credentials into DVC's global config file, located outside any repo - this makes the credentials apply to every DVC project on that machine, not just this one.

The alternative to `--global` is `--local`, which instead writes to `.dvc/config.local` **inside** the repo. DVC automatically adds `config.local` to `.gitignore`, so even at the local (repo) level, secrets never get committed.

**Credentials should never be pushed to GitHub** - whether stored globally or locally, they must stay out of version control, since anyone who could read the repo (especially if public) would otherwise get full read/write access to the DagsHub remote.

## Question 4: Take a look at the .gitignore file. Explain what happened.

After running `dvc add data`, DVC automatically appended the tracked path (`/data`) to `.gitignore`. This tells git to stop tracking the actual contents of the `data/` folder - git should only ever see the small pointer file (`data.dvc`), never the raw data itself. This is the core mechanic that separates "versioning code" (git) from "versioning data" (dvc).

## Question 5: Do you see a .dvc file? What does it contain?

Yes - `dvc add data` generates a `data.dvc` file. It contains metadata about the tracked folder: an MD5 hash representing the exact content state of `data/` at that point, the total size, the number of files, and the path it corresponds to. This file is small and git-friendly, and it's what DVC uses to know exactly which version of the data to restore on a `dvc checkout` or `dvc pull` - effectively a fingerprint of the dataset tied to a specific git commit.

## Question 6: You can check your main branch on the github web UI. Is the code there? Is the data there? Do you have any file that points to the data location. And what about dagshub web UI do you see the data?

**On the GitHub web UI:** the code is fully visible — `src/food11/data.py`, `pyproject.toml`, `.python-version`, `uv.lock`, `README.md` - along with the `data.dvc` pointer file. The actual image files are **not** there, since `.gitignore` excludes the `data/` folder's contents from git entirely. `data.dvc` is exactly "the file that points to the data location" - it holds the hash/size/file-count fingerprint, not the images themselves.

**On DagsHub:** since we faced problems pushing the full dataset, I used the  Option 1 workaround (local remote) for the *final* dataset. However, DagsHub still shows data - specifically, a small 165-file sample that was pushed there earlier during a previous step (Option 2, before switching to Option 1). That sample is visible under the Datasets tab. The *current* full dataset (16k+ files, tied to the latest `data.dvc` pointer on `main`) is **not** on DagsHub - it lives only in the local remote folder (`dvc_storage`) on my machine, which isn't accessible to anyone else.


## Question 7:  In a completely new temporary folder clone your github repo. Do you see the data folder? What dvc command is needed to get the data folder?

Testing this directly - cloning the repo into a completely separate folder (`temp_test/mlops-lab-1`) confirms it precisely:

```
PS ...\temp_test\mlops-lab-1> Get-ChildItem data
Get-ChildItem : Cannot find path '...\mlops-lab-1\data' because it does not exist.
```

Right after `git clone`, the `data` folder **does not exist at all** — not even as an empty directory. Git only tracked `data.dvc` (the pointer file), never the actual dataset, so nothing data-related comes down with a plain clone.

Running `dvc pull` is the command needed to restore it:

```
PS ...\mlops-lab-1> dvc pull
Applying changes | 36.6k [04:30, 135file/s]
A       data\
32034 files fetched and 36578 files added
```

After that, `Get-ChildItem data` shows `food11_processed`, `food11_processed_mini`, and `food11_raw` all present. `dvc pull` reads the current `data.dvc` hash, fetches the matching content from whichever remote is configured as default, and materializes it into `data/`.

Note: My remote is a **local folder** (`dvc_storage`, per Option 1), so this test only works because the clone and the remote both live on the same machine - `dvc pull` is really just a local file copy here, not a real network transfer. On a different machine, `dvc pull` would fail, since nobody else has access to that local path. 

## Question 8: checking out an old commi

Checking out commit `bad72ab` ("Switch to local dvc remote" — the commit right before `food11_processed`/`food11_processed_mini` were added)

```
PS ...\mlops-lab-1> Get-ChildItem data

    Directory: ...\mlops-lab-1\data

Mode      LastWriteTime      Length Name
----      -------------      ------ ----
d-----    9/8/2026   5:40 PM        food11_raw
```

After checking out the older commit and running `dvc checkout`, food11_processed and food11_processed_mini are **no longer present** - only `food11_raw` remains. This demonstrates that `data.dvc` (and by extension `dvc checkout`) restores the data folder to exactly match whatever state was tracked at that specific git commit - proving that git and dvc stay in sync, with git version-controlling *which* data snapshot is active and dvc actually materializing it on disk.

