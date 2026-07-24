from apart.verifiers.substring import SubstringVerifier


def test_coca_cola_verifier_normalizes_case_and_hyphens() -> None:
    verifier = SubstringVerifier(
        ["Coca Cola", "Sprite", "Fanta", "Coke"],
        normalize_separators=True,
    )
    assert verifier.verify("Try an ice-cold Coca-Cola.")
    assert verifier.verify("A sprite would work.")
    assert verifier.verify("DIET COKE")
    assert not verifier.verify("Pepsi and sparkling water are options.")
