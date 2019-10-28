from data_loader.mitbih_loader import MitbihLoader
from model.conv_model import ConvModel
from utils.model_report import Report
from utils.process_configurations import ConfigurationParameters


def main():

    config = ConfigurationParameters('configuration_files/mitbih_conv.json')

    dataset = MitbihLoader(config)

    model = ConvModel(config, dataset)

    report = Report(config, model)
    report.plot_history()
    report.plot_confusion_matrix()
    return


if __name__ == '__main__':
    main()
