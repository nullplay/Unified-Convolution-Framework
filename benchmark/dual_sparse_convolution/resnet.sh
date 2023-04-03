i=0
while read cfg; do 
  ((i+=1));
  echo -n Layer c$i " / " ;
  echo -n " ";
  ./dual_sparse_2dconv $cfg $2 NPQM NHWC DDDS RSCM DDDS npqrscm pq 20 4 bench c$i ; 
done < $1 
