
POLICY_CONFIG=./config/policy/motivo_23dof.yaml
MODEL_ONNX_PATH=./results/bfmzero-isaac-23dof-new/exported/FBcprAuxModel.onnx
TASK=./config/exp/teleop/locomotion.yaml

python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1_23dof.yaml \
    --policy_config ${POLICY_CONFIG} \
    --model_path ${MODEL_ONNX_PATH} \
    --task  ${TASK}
