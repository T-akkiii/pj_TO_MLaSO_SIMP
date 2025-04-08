import pytest
import numpy as np
import torch

@pytest.fixture
def sample_mesh():
    # テスト用の単純なメッシュデータを生成
    return {...}

@pytest.fixture
def mock_freefem():
    # FreeFEMのモックオブジェクト
    return {...}

@pytest.fixture
def optimization_params():
    # 最適化パラメータの設定
    return {...} 