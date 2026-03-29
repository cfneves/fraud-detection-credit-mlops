"""
data_download.py
----------------
Download de datasets do Kaggle via API.

Uso:
    python src/data_download.py --dataset fraud
    python src/data_download.py --dataset ieee
    python src/data_download.py --dataset credit_scoring
    python src/data_download.py --all

Pré-requisito:
    ~/.kaggle/kaggle.json com suas credenciais
    pip install kaggle
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:  # noqa: E501
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_kaggle_api():
    """Verifica se as credenciais do Kaggle estão configuradas."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("=" * 60)
        print("KAGGLE API NÃO CONFIGURADA")
        print("=" * 60)
        print("\nPassos para configurar:")
        print("1. Acesse https://www.kaggle.com/settings")
        print("2. Clique em 'Create New Token'")
        print("3. Salve o kaggle.json em ~/.kaggle/")
        print("4. Execute: chmod 600 ~/.kaggle/kaggle.json")
        print("\nOu defina as variáveis de ambiente:")
        print("   export KAGGLE_USERNAME=seu_usuario")
        print("   export KAGGLE_KEY=sua_chave")
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Erro ao autenticar Kaggle API: {e}")
        sys.exit(1)


def download_dataset(api, dataset_id: str, output_path: str, dataset_type: str = "dataset"):
    """Baixa e extrai um dataset do Kaggle."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaixando: {dataset_id}")
    print(f"Destino: {output_dir}")

    try:
        if dataset_type == "competition":
            api.competition_download_files(dataset_id, path=str(output_dir))
        else:
            api.dataset_download_files(dataset_id, path=str(output_dir), unzip=True)

        print(f"Download concluído.")

        # Listar arquivos baixados
        files = list(output_dir.glob("*"))
        print(f"Arquivos disponíveis:")
        for f in files:
            size = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size:.1f} MB)")

    except Exception as e:
        print(f"Erro no download: {e}")
        print("\nVerifique se você aceitou as regras da competição/dataset no Kaggle.")


def download_fraud(api, config: dict):
    """Dataset Credit Card Fraud Detection."""
    dataset_id = config["data"]["kaggle_datasets"]["fraud"]
    output_path = Path(config["data"]["raw_path"]) / "fraud"
    download_dataset(api, dataset_id, str(output_path))


def download_ieee(api, config: dict):
    """Dataset IEEE-CIS Fraud Detection (competição)."""
    # IEEE é uma competição — requer aceitar as regras
    dataset_id = "ieee-fraud-detection"
    output_path = Path(config["data"]["raw_path"]) / "ieee"
    print("\nATENÇÃO: O dataset IEEE-CIS é uma competição do Kaggle.")
    print("Você precisa aceitar as regras em:")
    print("https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
    download_dataset(api, dataset_id, str(output_path), dataset_type="competition")


def download_credit_scoring(api, config: dict):
    """Dataset Give Me Some Credit."""
    dataset_id = config["data"]["kaggle_datasets"]["credit_scoring"]
    output_path = Path(config["data"]["raw_path"]) / "credit_scoring"
    download_dataset(api, dataset_id, str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Download de datasets via Kaggle API")
    parser.add_argument(
        "--dataset",
        choices=["fraud", "ieee", "credit_scoring"],
        help="Dataset específico para baixar"
    )
    parser.add_argument("--all", action="store_true", help="Baixar todos os datasets")
    parser.add_argument("--config", default="config/config.yaml", help="Caminho do arquivo de config")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)
    api = setup_kaggle_api()

    if args.all or args.dataset == "fraud":
        download_fraud(api, config)

    if args.all or args.dataset == "ieee":
        download_ieee(api, config)

    if args.all or args.dataset == "credit_scoring":
        download_credit_scoring(api, config)

    print("\nDownload(s) concluído(s).")
    print(f"Dados salvos em: {config['data']['raw_path']}/")


if __name__ == "__main__":
    main()
