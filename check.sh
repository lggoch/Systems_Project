# Getting all pods
PODS=$(kubectl get pods --field-selector=status.phase==Running -o=jsonpath='{range .items[*]}{.metadata.name}{","}')
IFS=',' read -ra ADDR <<< "$PODS"

while true; do
  for pod in "${ADDR[@]}"; do

    {
      kubectl cp $pod:/worker/weights.pickle ./weights_2.pickle
    } &> /dev/null

    if [ $? -eq 0 ]; then
      echo $pod has converged! The weights are available locally in weights.pickle!
      kubectl delete -f stateful_set.yaml
      exit 1
    fi

  done
done
