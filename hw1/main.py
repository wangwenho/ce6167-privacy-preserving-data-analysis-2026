from centralized_learning import main as cl_main
from federated_learning import main as fl_main


def main():
    print("Running CL...")
    cl_main()

    print("Running FL...")
    fl_main()


if __name__ == "__main__":
    main()
