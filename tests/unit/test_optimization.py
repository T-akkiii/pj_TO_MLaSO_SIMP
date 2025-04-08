def test_objective_function():
    """目的関数の基本的な性質をテスト"""
    # 既知の解析解との比較
    expected = ...
    result = calculate_objective(...)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_gradient_calculation():
    """感度解析の検証"""
    # 数値微分との比較
    numerical_grad = ...
    analytical_grad = ...
    np.testing.assert_allclose(numerical_grad, analytical_grad, rtol=1e-4) 