MASTER_HOST=104.171.203.218
WORKERS=152.67.250.123,192.18.129.244

mpirun -np 3 \
-H $HOST,$WORKERS \
-x MASTER_ADDR=$MASTER_HOST \
-x MASTER_PORT=1234 \
-x HF_AUTH_TOKEN=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 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 train.py -d "soul11zz/image-caption-desc-only" -m "soul11zz/image-caption-desc-only" -b 8 -e 1 --test-best --metric meteor
