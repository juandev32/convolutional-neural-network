import pkg_resources
import subprocess
import sys
import tensorflow as tf

#ensure youre running cuda 11.8 if you want to enable gpu for faster model training, optinal but faster training
def check_cuda():
    try:
        print(tf.sysconfig.get_build_info()["cuda_version"])
    except KeyError as k:
        print(f"Cuda version is not recognized by tensorflow, probably cuda version >11.8")
    except Exception as e:
        print(f"Error has occured while trying to get the CUDA version {str(e)}")

def validate_requirements(requirments_file):
    #pkg_resources is better than storing via import_module, pillow isnt present as import but required and present in pip list
    installed_packages={pkg.key for pkg in pkg_resources.working_set}

    #ensure all the dependencies in the requiremnets.txt file are installed
    try:
        with open(requirments_file,'r') as reqfile:
            #read lines from requirement file
            for line in reqfile:
                
                package = line.split('==')[0].strip()
                
                if package.lower() not in installed_packages:
                
                    while True:
                        promptInstall= input(f"{package} is not installed. Do you want to install it Y/N : ").strip().lower()
                        if promptInstall in ['y',"yes","ok","ye"]:
                            print(f"Installing package {package} ...")
                            #installs the missing package
                            subprocess.check_call([sys.executable,'-m','pip','install',package])
                            print(f"{package} has been installed")
                            break
                        elif promptInstall in ['n',"no","nope","on"]:
                            print(f"{package} is listed in the requirements file and needed to train the neural network, later exceptions may occur.")
                            break
                        else:
                            print("Enter a valid input 'Y' or 'N'.")
    except ImportError:
        print(package)         
    except FileNotFoundError:
        print("requirements.txt file not found. \nDo not move the file from the Multilayer Perceptron - Diabetes predictor Parent directory")
    except Exception as e:
        print(f'An exception as occured: {str(e)}')
    finally:
        print("All depencencies are installed with the correct version.")