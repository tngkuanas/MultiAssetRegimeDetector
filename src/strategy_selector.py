from collections import Counter

def get_portfolio_composition(symbols: list, asset_definitions: dict) -> dict:
    """
    Analyzes the composition of a portfolio based on predefined asset classes.

    :param symbols: A list of asset tickers in the portfolio.
    :param asset_definitions: A dictionary mapping tickers to asset classes.
    :return: A dictionary mapping asset classes to their percentage in the portfolio.
    """
    if not symbols:
        return {}

    # Use 'UNKNOWN' for any ticker not in the definitions map
    asset_classes = [asset_definitions.get(s, 'UNKNOWN') for s in symbols]
    
    # Count occurrences of each asset class
    class_counts = Counter(asset_classes)
    
    # Calculate percentage for each class
    total_assets = len(symbols)
    composition = {cls: (count / total_assets) * 100 for cls, count in class_counts.items()}
    
    return composition

def select_strategy_version(portfolio_composition: dict, selection_rules: list) -> str:
    """
    Selects the appropriate strategy version based on a set of rules and portfolio composition.

    :param portfolio_composition: A dictionary mapping asset classes to their percentage.
    :param selection_rules: A list of rules, where each rule has conditions and a version name.
    :return: The name of the selected strategy version.
    """
    # Sort rules by priority. Lower number = higher priority.
    rules = sorted(selection_rules, key=lambda r: r.get('priority', 999))

    for rule in rules:
        conditions = rule.get('conditions', {})
        
        # If there are no conditions, it's a default/fallback rule
        if not conditions:
            print(f"Selector: Matched default rule '{rule['name']}'.")
            return rule['version_to_use']

        is_match = True
        
        # --- Process all conditions for the rule ---
        
        # Condition 1: min_percentage_asset_class
        min_percentages = conditions.get('min_percentage_asset_class')
        if min_percentages:
            for asset_class, min_perc in min_percentages.items():
                if portfolio_composition.get(asset_class, 0) < min_perc:
                    is_match = False
                    break
            if not is_match: continue # Rule failed, try next rule
        
        # Condition 2: min_percentage_sum
        min_sum = conditions.get('min_percentage_sum')
        if min_sum:
            sum_classes = min_sum.get('classes', [])
            required_perc = min_sum.get('percentage', 101) # Default to an impossible value
            
            total_perc = sum(portfolio_composition.get(c, 0) for c in sum_classes)
            
            if total_perc < required_perc:
                is_match = False
            if not is_match: continue # Rule failed, try next rule

        # If we get here, it means all conditions for the current rule passed.
        if is_match:
            print(f"Selector: Matched rule '{rule['name']}'.")
            return rule['version_to_use']

    # As a final fallback in case rules are misconfigured or no rule matches
    print("Selector: No specific rule matched. Falling back to 'default_balanced'.")
    return 'default_balanced'


if __name__ == '__main__':
    # --- Example Usage ---
    # 1. Load mock config data (in a real scenario, this comes from config.yaml)
    mock_asset_defs = {
        'VOO': 'EQUITY_US_MARKET', 'VXUS': 'EQUITY_INTL_MARKET', 'GLD': 'COMMODITY_GOLD',
        'BND': 'BOND_US_AGG', 'GOOG': 'EQUITY_SINGLE_GROWTH', 'MSFT': 'EQUITY_SINGLE_GROWTH',
        'TLT': 'BOND_US_TREASURY_LONG', 'IEF': 'BOND_US_TREASURY_INTERMEDIATE'
    }
    mock_rules = [
        {
            'name': "Single Growth Stock Rule",
            'priority': 2,
            'conditions': {'min_percentage_asset_class': {'EQUITY_SINGLE_GROWTH': 20}},
            'version_to_use': "single_stock_growth"
        },
        {
            'name': "Bond Heavy Rule",
            'priority': 3,
            'conditions': {
                'min_percentage_sum': {
                    'classes': ['BOND_US_AGG', 'BOND_US_TREASURY_LONG', 'BOND_US_TREASURY_INTERMEDIATE'],
                    'percentage': 40
                }
            },
            'version_to_use': "bond_heavy"
        },
        {
            'name': "Default Balanced Rule",
            'priority': 99,
            'conditions': {},
            'version_to_use': "default_balanced"
        }
    ]

    # 2. Test with a growth stock portfolio
    print("--- Testing Growth Portfolio ---")
    growth_portfolio = ['VOO', 'GLD', 'BND', 'GOOG', 'MSFT'] # 40% SINGLE_GROWTH
    growth_composition = get_portfolio_composition(growth_portfolio, mock_asset_defs)
    print(f"Portfolio: {growth_portfolio}")
    print(f"Composition: {growth_composition}")
    selected_version_growth = select_strategy_version(growth_composition, mock_rules)
    print(f"Selected Version: {selected_version_growth}\n") # Expected: single_stock_growth

    # 3. Test with a bond-heavy portfolio
    print("--- Testing Bond Portfolio ---")
    bond_portfolio = ['VOO', 'VXUS', 'BND', 'TLT', 'IEF'] # 60% BONDS
    bond_composition = get_portfolio_composition(bond_portfolio, mock_asset_defs)
    print(f"Portfolio: {bond_portfolio}")
    print(f"Composition: {bond_composition}")
    selected_version_bond = select_strategy_version(bond_composition, mock_rules)
    print(f"Selected Version: {selected_version_bond}\n") # Expected: bond_heavy

    # 4. Test with a balanced portfolio (matches default)
    print("--- Testing Balanced Portfolio ---")
    balanced_portfolio = ['VOO', 'VXUS', 'GLD', 'BND']
    balanced_composition = get_portfolio_composition(balanced_portfolio, mock_asset_defs)
    print(f"Portfolio: {balanced_portfolio}")
    print(f"Composition: {balanced_composition}")
    selected_version_balanced = select_strategy_version(balanced_composition, mock_rules)
    print(f"Selected Version: {selected_version_balanced}\n") # Expected: default_balanced
