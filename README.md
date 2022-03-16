# compare_mat_rank

main program: get_emb.py  
To successfully run the program, phraser.pkl needs to be added in this directory, and embedding files should be put in "models" directory.  
def process_file: convert zt/mpf content into a dict with material name as key and their rank as value.  
To use either kind of embedding, comment out the other type.  
To calculate correlation of either rank (zt/mpf) and rank from embedding, use zt_rank or pf_rank and comment out the other type.  
  
Current result:   
output embedding: zt: 0.32, mpf: 0.53  
word embedding: zt: 0.41, mpf: 0.58  
