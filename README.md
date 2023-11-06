# Regression Analysis on Cloud Services for IaaS PaaS & CaaS

## Description

py_regression is a comprehensive toolkit for performing regression analysis specifically tailored for Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Container as a Service (CaaS) cloud services. It equips users with the ability to analyze trends, predict performance, and understand key factors affecting cloud service metrics. This project provides Python scripts, Jupyter notebooks, and data visualization tools for in-depth regression analysis.


## Installation

Clone the project repository and set up a virtual environment:

``` bash
git clone https://github.com/yourusername/py_regression.git
cd py_regression
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Navigate to the project directory and run the provided Python scripts or Jupyter notebooks.

Execute a script: (cloud models are iaas, caas, paas)
```bash
python3 {cloudmodel}_reg.py
```
Launch a Jupyter notebook:
``` bash
jupyter notebook <notebook_name>.ipynb
```
Interact with the analysis by modifying parameters or input data as needed.
## Dependencies

A list of Python libraries upon which the project depends is available in requirements.txt, including:

* NumPy for numerical computing
* pandas for data manipulation
* Matplotlib for plotting
* scikit-learn for machine learning algorithms


## License

This project is open-sourced under the MIT License. See the LICENSE file for more information.


## Contributing
If you would like to contribute to the development of py_regression, please fork the repository and submit a pull request with your proposed changes.

## Support
For queries or technical support, please open an issue on the GitHub issues page of the repository.



<!-- Additional sections: (Optional) Any other sections that you want to include, such as a section on known issues, contributing, or contacting you for support. -->








<!-- # py_regression
Regression analysis

## Virtual enviroment
### Create
```bash 
python -m venv regenv  
```

### Enable
```bash
source ../regenv/bin/activate
```


## Install sickit scipy matplotlib
```bash
pip install -U scikit-learn scipy matplotlib -->
```
