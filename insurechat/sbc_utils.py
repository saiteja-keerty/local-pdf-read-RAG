import re

def parse_plan_terms(text: str) -> dict:
    """Extract common plan numeric terms from SBC text.

    Returns a dict with keys like 'overall_deductible_network_individual',
    'out_of_pocket_limit_network_individual', 'pcp_copay', 'specialist_copay',
    'urgent_copay', 'hospital_coinsurance', 'other_coinsurance'. Values are numbers.
    """
    terms = {}
    # overall deductible (network) individual
    m = re.search(r"For network providers\s*\$\s?([0-9,]+)\s*individual", text, re.I)
    if m:
        terms['overall_deductible_network_individual'] = float(m.group(1).replace(',', ''))
    else:
        m2 = re.search(r"deductible[^\$]{0,60}\$\s?([0-9,]+)", text, re.I)
        if m2:
            terms['overall_deductible_network_individual'] = float(m2.group(1).replace(',', ''))

    # out-of-pocket limit
    m = re.search(r"For network providers\s*\$\s?([0-9,]+)\s*individual\s*/\s*\$\s?([0-9,]+)\s*family", text, re.I)
    if m:
        terms['out_of_pocket_limit_network_individual'] = float(m.group(1).replace(',', ''))
    else:
        m2 = re.search(r"out-of-pocket limit[\s\S]{0,80}\$\s?([0-9,]+)\s*individual", text, re.I)
        if m2:
            terms['out_of_pocket_limit_network_individual'] = float(m2.group(1).replace(',', ''))

    # copays
    m = re.search(r"Primary care visit[\s\S]{0,80}\$\s?([0-9,]+)", text, re.I)
    if m:
        terms['pcp_copay'] = float(m.group(1).replace(',', ''))
    m = re.search(r"Specialist\s*Visit[\s\S]{0,80}\$\s?([0-9,]+)", text, re.I)
    if m:
        terms['specialist_copay'] = float(m.group(1).replace(',', ''))
    m = re.search(r"Urgent care[\s\S]{0,80}\$\s?([0-9,]+)", text, re.I)
    if m:
        terms['urgent_copay'] = float(m.group(1).replace(',', ''))

    # coinsurance selection by nearby context
    for mm in re.finditer(r"([0-9]{1,3})%\s*(?:\n|\s)*Coinsurance", text, re.I):
        pct = float(mm.group(1)) / 100.0
        head = text[max(0, mm.start()-80):mm.start()].lower()
        if any(k in head for k in ('hospital', 'facility', 'hospital (facility)', 'facility fee')):
            terms['hospital_coinsurance'] = pct
            break
    if 'hospital_coinsurance' not in terms:
        for mm in re.finditer(r"([0-9]{1,3})%\s*(?:\n|\s)*Coinsurance", text, re.I):
            pct = float(mm.group(1)) / 100.0
            head = text[max(0, mm.start()-80):mm.start()].lower()
            if 'other' in head:
                terms['other_coinsurance'] = pct
                break

    # fallback: any coinsurance
    if 'hospital_coinsurance' not in terms and 'other_coinsurance' not in terms:
        m = re.search(r"([0-9]{1,3})%\s*Coinsurance", text, re.I)
        if m:
            terms['other_coinsurance'] = float(m.group(1)) / 100.0

    return terms


def estimate_member_payment(bill_amount: float, service_type: str, network: str, plan: dict) -> str:
    """Estimate member payment for a single service given plan terms.

    Simplified rules:
    - Member pays deductible first up to overall deductible
    - After deductible, coinsurance applies to remaining amount
    - Cap at out-of-pocket limit if available
    """
    ded = plan.get('overall_deductible_network_individual', 0.0)
    oop = plan.get('out_of_pocket_limit_network_individual', None)
    if service_type == 'hospital':
        coin = plan.get('hospital_coinsurance', plan.get('other_coinsurance', 0.0))
    else:
        coin = plan.get('other_coinsurance', 0.0)

    # member pays up to deductible first
    member_ded = min(ded, bill_amount)
    remaining = max(0.0, bill_amount - member_ded)
    member_after_ded = coin * remaining
    member_total = member_ded + member_after_ded

    if oop is not None:
        member_total_capped = min(member_total, oop)
    else:
        member_total_capped = member_total

    return f"Estimate for ${bill_amount:,.0f} {('in-network' if network=='network' else '')} {service_type} bill: member pays ${member_total_capped:,.2f} (raw calc ${member_total:,.2f})"
