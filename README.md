# Top-K Search and Unlearning with LSH

When you clone this repo, please use `git clone --recurse-submodules`.
Alternatively, you can manually `git clone caboose` or `git clone caboose_index`
into the repo.

## Installing Caboose
Make sure you have Python 3.9 and Rust installed: https://www.rust-lang.org/tools/install

We recommend creating a separate virtual env for testing [Caboose](https://deem.berlin/pdf/caboose.pdf):
```
conda create --name caboose python=3.9
conda activate caboose
```

Once you `cd` into the `caboose` directory,
 * Install Cython (needed for similaripy) with `pip install Cython==0.29.32`
 * Install the dependencies with `pip install -r requirements.txt`
 * Build the project with `maturin develop --release`

Whenever running code from `caboose`, make sure `conda activate caboose`.
Once done with your `caboose` workflow, `conda deactivate`.