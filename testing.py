from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.analytical import BlackScholesEngine
# Define a contract
contract = OptionContract(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.2,
    option=OptionType.CALL,
    style=ExerciseStyle.EUROPEAN
)
# Price it
engine = BlackScholesEngine()
result = engine.price(contract)
print(f"Price: {result.price}")