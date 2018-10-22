
# Getting all pods
PODS=$(kubectl get pods --field-selector=status.phase==Running -o=jsonpath='{range .items[*]}{.metadata.name}{","}')
IFS=',' read -ra ADDR <<< "$PODS"

for pod in "${ADDR[@]}"; do

  cmd="screen -r $pod"
  prefix='tell app "Terminal" to do script "'
  suffix='"'
  cmd="$prefix $cmd $suffix"
  tmp="'"
  cmd="osascript -e $tmp$cmd$tmp"
  eval $cmd

done
