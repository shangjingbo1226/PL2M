./../bin/pl2m_train pcd.conf ../data/userFeatMat ../data/itemFeatMat ../data/train ../data/test models/model

for i in `seq 1 40`;
do
    ./../bin/pl2m_infer ../data/test models/model.$i pred.txt
done

