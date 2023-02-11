mpirun -np 8 \
-H 104.171.203.48:8 \
-x MASTER_ADDR=104.171.203.48 \
-x MASTER_PORT=1234 \
-x HF_AUTH_TOKEN=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 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 train.py -d "soul11zz/image-caption-desc-only" -m "soul11zz/image-caption-desc-only" -b 8 -e 1 --test-best --metric meteor
