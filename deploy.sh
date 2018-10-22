killall SCREEN

# Default number of workers is 4
mode=${1:-"Async"}
nb=${2:-4}

echo Terminating Containers..
kubectl delete -f stateful_set.yaml

if [ "$mode" != "Async" ]; then
  worker_type="Sync_Worker"
  nb=$(echo $(( nb + 1 ))) 
else
  worker_type="Async_Worker" 
fi

echo Launching with $nb workers
kubectl create -f stateful_set.yaml
json={'"spec"':{'"replicas"':$nb}}
kubectl patch statefulset systems -p "$json"

echo Creating Containers..
while [ ${#ADDR[@]} != $nb ]; do
PODS=$(kubectl get pods --field-selector=status.phase==Running -o=jsonpath='{range .items[*]}{.metadata.name}{","}')
IFS=',' read -ra ADDR <<< "$PODS"
done
# sleep 30

# Getting all pods
PODS=$(kubectl get pods --field-selector=status.phase==Running -o=jsonpath='{range .items[*]}{.metadata.name}{","}')
IFS=',' read -ra ADDR <<< "$PODS"

function join_by { local IFS="$1"; shift; echo "$*"; }

# For all pods
i=0
for pod in "${ADDR[@]}"; do

  # Get the hostname of all other pods
  other_hosts=()
  for other_pod in "${ADDR[@]}"; do
    if [ "$pod" != "$other_pod" ]; then
      other_hosts+=("$other_pod.nginx.cs449g3.svc.k8s.iccluster.epfl.ch")
    fi
  done

  # Running the worker

  other_hosts=$(join_by , "${other_hosts[@]}")

  # Split the data if there are more than 3 workers
  if [[ "$pod" == "${ADDR[$nb-1]}" && "$mode" != "Async" ]]; then
    echo Coordinator created
    screen -dmS $pod bash -c "kubectl exec -it $pod python dist_SVM.py Coordinator ${other_hosts} 100; exec sh"

  else

    if [ $nb \> 4 ]; then
      i=$(echo $(( (i + 1)%4 )))
      screen -dmS $pod bash -c "kubectl exec -it $pod python dist_SVM.py $worker_type ${other_hosts} 100 $i; exec sh"
    else
      screen -dmS $pod bash -c "kubectl exec -it $pod python dist_SVM.py $worker_type ${other_hosts} 100; exec sh"
    fi

  fi

    


done

echo
echo Pour afficher les workers:
echo bash show.sh
echo
echo Pour les kill:
echo "kubectl delete -f stateful_set.yaml"
bash check.sh
