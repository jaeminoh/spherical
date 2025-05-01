from data_summary import load_SAbDab
import os

d = load_SAbDab()

os.system("sh relax_score.sh")