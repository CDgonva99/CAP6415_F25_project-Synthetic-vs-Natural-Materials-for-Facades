PY?=python
CFG_DIR=configs
train-nat:
	$(PY) src/train.py --config $(CFG_DIR)/train_nat.yaml

eval:
	$(PY) src/train.py --config $(CFG_DIR)/eval.yaml --eval

td-demo:
	$(PY) src/infer_td.py --config $(CFG_DIR)/eval.yaml --osc_port 8000
