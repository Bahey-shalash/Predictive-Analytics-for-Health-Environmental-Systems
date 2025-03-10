#!/bin/bash

main() {
	if [[ "${VIRTUAL_ENV-x}" == "x" ]]; then
		echo "INFO: activating venv"
		if [[ ! -x "${HOME}/Desktop/myfiles/ENG209_2024Fall/venv/bin/activate" ]]; then
			echo "ERROR: ${HOME}/Desktop/myfiles/ENG209_2024Fall/venv does not exist"
			return 1
		fi
		. "${HOME}/Desktop/myfiles/ENG209_2024Fall/venv/bin/activate"
	fi

        pip install --no-input plotly scikit-learn nbformat pandas numpy matplotlib jupyterlab-latex
	pushd ${HOME}/Desktop/myfiles/ENG209_2024Fall/eng209-2024-final/notebooks
	pip install --no-input -e eng209
	popd
}

main "${@}"
