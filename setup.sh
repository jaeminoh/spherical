# requirements
uv sync 

# Rosetta script
git clone https://github.com/RosettaCommons/rosetta
cd rosetta/source
./scons.py -j<NumOfJobs> mode=release bin