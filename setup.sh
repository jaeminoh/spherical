# requirements
uv sync 
source .venv/bin/activate
pre-commit install

# Rosetta script
git clone https://github.com/RosettaCommons/rosetta
cd rosetta/source
./scons.py -j200 mode=release bin # modify 200 to your CPU cores
cd ..
git submodule update --init -- ./tools/