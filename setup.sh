# requirements
uv sync 
source .venv/bin/activate
# Rosetta script
git clone https://github.com/RosettaCommons/rosetta
cd rosetta/source
./scons.py -j200 mode=release bin
cd ..
git submodule update --init -- ./tools/
