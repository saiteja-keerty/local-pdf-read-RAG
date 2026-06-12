# Medical insurance terms for newcomers

This file is a local knowledge base for InsureChat. It explains common U.S. medical insurance terms in plain language for people who may be unfamiliar with the U.S. health care system. Sources used: HealthCare.gov glossary pages for deductible, copayment, coinsurance, out-of-pocket maximum, and allowed amount.

## Quick mental model

In the U.S., medical care often has three separate prices:

1. The provider's billed charge: what the hospital, doctor, lab, or pharmacy asks for.
2. The allowed amount: the price the insurance plan recognizes for a covered service.
3. The patient responsibility: what the patient may owe after insurance rules are applied.

Insurance does not always pay the full bill. It usually follows plan rules: deductible first, then copay or coinsurance, up to an out-of-pocket maximum for covered in-network care.

## Term table

| Term | Plain-language meaning | What to look for on bills, EOBs, or claims | Example |
|---|---|---|---|
| Premium | The regular amount paid to keep insurance active, usually monthly. | Usually not on a medical bill; appears in plan or account billing. | You pay a monthly premium even if you do not visit a doctor. |
| Deductible | The amount you pay for covered services before the plan starts paying many costs. | "Deductible", "applied to deductible", "deductible remaining". | If the deductible is $2,000, you may pay covered costs until $2,000 is reached. |
| Copayment / copay | A fixed dollar amount for a covered service. | "Copay", "office visit copay", "ER copay", "Rx copay". | A primary-care visit might have a $25 copay. |
| Coinsurance | A percentage of the allowed amount that you pay, usually after deductible. | "Coinsurance", "member coinsurance", "patient coinsurance". | If allowed amount is $100 and coinsurance is 20%, patient pays $20. |
| Out-of-pocket maximum | The yearly cap on what you pay for covered in-network services through deductible, copay, and coinsurance. | "OOP max", "out-of-pocket", "accumulator", "remaining". | After reaching the max, covered in-network benefits are generally paid 100% by the plan. |
| Allowed amount | The maximum amount the plan recognizes for a covered service; also called allowed charge, payment allowance, eligible expense, or negotiated rate. | "Allowed", "eligible", "negotiated", "plan discount". | Provider bills $300, allowed amount is $180; cost sharing is usually based on $180. |
| Discount / adjustment | The part of a provider's billed charge removed because of an insurer's negotiated rate or another billing adjustment. It is usually not an amount the patient pays. | "Discount", "adjustment", "network savings", "provider discount". | If the provider bills $300 and the allowed amount is $180, the discount or adjustment may be $120. |
| Provider charge / billed amount | The amount the doctor, hospital, lab, or pharmacy billed before plan discounts. | "Amount billed", "charge", "provider billed". | A hospital may bill $1,000 before discounts. |
| Plan paid | The amount the insurer paid to the provider or member. | "Plan paid", "insurance paid", "paid by plan". | If allowed is $180 and patient owes $40, plan may pay $140. |
| Patient responsibility | The amount the patient may owe after insurance processing. | "You owe", "patient responsibility", "member responsibility", "amount due". | This should generally match deductible + copay + coinsurance + non-covered amounts. |
| Explanation of Benefits (EOB) | A statement from insurance explaining how a claim was processed. It is not always a bill. | "This is not a bill", claim number, service date, provider, billed, allowed, paid, you owe. | Use the EOB to compare against a provider bill. |
| Claim | A request for payment sent to insurance for a medical service. | "Claim number", "claim status", "processed date". | A provider sends a claim after your visit. |
| In-network | A provider contracted with the insurance plan, usually at negotiated prices. | "Network", "participating provider", "preferred provider". | In-network care is usually cheaper. |
| Out-of-network | A provider not contracted with the plan; patient cost may be higher. | "Out of network", "non-participating". | Out-of-network charges may not count toward in-network limits. |
| Balance billing | When a provider bills the difference between its charge and the allowed amount. | "Balance", "above allowed amount", "not covered by plan". | If billed is $500 and allowed is $300, balance billing may be $200 when permitted. |
| Prior authorization | Approval required by the plan before some services. | "Authorization", "preauthorization", "approval required". | MRI or surgery may require prior authorization. |
| Referral | A direction from a primary-care doctor to see another provider, often a specialist. | "Referral required", "PCP referral". | Some HMO-style plans require referrals. |
| Formulary | The plan's list of covered drugs. | "Drug tier", "formulary", "preferred drug". | A generic drug may be cheaper than a brand drug. |
| CPT code | A procedure/service code used in medical billing. | Five-digit codes on bills or claims. | 99213 can indicate an office visit level. |
| ICD-10 code | A diagnosis code explaining the medical reason for care. | Codes such as E11.9 or J02.9. | Used to support medical necessity. |
| Denial | Insurance did not approve payment for all or part of a claim. | "Denied", "not covered", "reason code". | Denials may be appealable. |
| Appeal | A request for the insurer to review a denial or payment decision. | "Appeal rights", "how to appeal", deadline. | Appeal if records show the service should be covered. |
| Coordination of benefits | Rules for which insurer pays first when someone has more than one plan. | "Primary", "secondary", "COB". | One plan pays first, the second may pay some remaining amount. |
| Fixed indemnity insurance | A supplemental plan that pays a set dollar amount for covered events, not necessarily based on the full bill. | "Fixed benefit", "indemnity", "we pay per day/per visit". | It may pay $100 for a visit even if the bill is $72 or $175. |

## How to explain a bill or claim

When a user uploads or describes a medical bill, EOB, or claim:

1. Identify whether the document is a provider bill, an EOB, a plan summary, or a claim status page.
2. Extract service date, provider, billed amount, allowed amount, plan paid, adjustments/discounts, deductible, copay, coinsurance, non-covered charges, and patient responsibility.
3. Explain the math in simple steps.
4. Flag mismatches, such as provider bill amount being higher than EOB patient responsibility.
5. Suggest safe next actions: compare EOB with bill, call insurer/provider using the phone number on the card or bill, ask about coding, network status, prior authorization, appeal rights, and payment plans.

## Safety rules

- Do not claim the user definitely owes or does not owe money unless the document clearly says so.
- Do not provide legal, tax, or medical advice.
- For non-U.S. users, explain that U.S. insurance terms are plan-specific and may not match their home country's system.
- Prefer document evidence over general knowledge.
- If a calculation is uncertain, show assumptions and ask for the missing value.
