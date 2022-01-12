SRC=cifar100_fsl.py

for C in 4 8 16
do
    for A in resnet50 densenet121 resnet506
    do
        for L in 1 2
        do
            python $SRC --cloud_arch $A --num_clients $C --pretrain ./weights/${A}_cifar10.pth --pretrain_layer $L --log --naive
            rc=$?; if [ $rc != 0 ]; then exit $rc; fi
            python $SRC --cloud_arch $A --num_clients $C --pretrain ./weights/${A}_cifar10.pth --pretrain_layer $L --log
            rc=$?; if [ $rc != 0 ]; then exit $rc; fi
        done
    done
done
