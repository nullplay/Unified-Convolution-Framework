i=0
while read cfg; do 
  ((i+=1));
  echo -n Layer c$i " / " ;
  echo -n " ";
  ./filter_sparse_2dconv_gpu $cfg 90 NMPQ NCHW UUUU MRSC UUUC bench c$i ; 
done < $1 
