# Conversion between Ito and Stratonovich Integrals

## Introduction
The difference between the **Ito integral** and the **Stratonovich integral** lies in the evaluation point of the stochastic term.

1. **Ito Integral:**  
   \[
   dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
   \]
   Here, the stochastic increment \( dW_t \) is evaluated at the left endpoint (previous time step).

2. **Stratonovich Integral:**  
   \[
   dX_t = \mu(X_t, t) dt + \sigma(X_t, t) \circ dW_t
   \]
   In this case, \( dW_t \) is evaluated at the **midpoint** between two time steps.

## Ito’s Lemma
Consider a function \( f(X_t, t) \), where \( X_t \) follows an Ito process:
   \[
   dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t
   \]
Applying **Ito’s Lemma**, the differential of \( f(X_t, t) \) is:
   \[
   df = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial X} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial X^2} \right) dt + \sigma \frac{\partial f}{\partial X} dW_t.
   \]
This additional **\(\frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial X^2} dt\)** term is unique to Ito calculus and does not appear in classical differentiation.

## Conversion Formula (Ito to Stratonovich)
The **Ito-to-Stratonovich** correction formula is:
   \[
   dX_t^{\text{Stratonovich}} = dX_t^{\text{Ito}} + \frac{1}{2} \sigma(X_t, t) \frac{\partial \sigma(X_t, t)}{\partial X} dt.
   \]
Expanding using Ito’s formula:
   \[
   dX_t^{\text{Stratonovich}} = \left[ \mu(X_t, t) + \frac{1}{2} \sigma(X_t, t) \frac{\partial \sigma(X_t, t)}{\partial X} \right] dt + \sigma(X_t, t) dW_t.
   \]
This shows that the drift term in the **Stratonovich integral** is modified by an additional correction term.

## Midpoint vs. Left-point Summation Rule
- **Ito Integral (Left-point rule)**:
  \[
  \sum_{i} f(X_i) (W_{i+1} - W_i)
  \]
  Here, the function \( f(X_i) \) is evaluated at the previous step.
  
- **Stratonovich Integral (Midpoint rule)**:
  \[
  \sum_{i} f\left(\frac{X_{i} + X_{i+1}}{2}\right) (W_{i+1} - W_i)
  \]
  The function is evaluated at the midpoint, which ensures that standard calculus rules (like the chain rule) hold.

## Conclusion
- **Ito integrals are non-anticipative**, meaning the next step depends only on past information.
- **Stratonovich integrals maintain classical calculus properties**, making them useful in physics and engineering.
- The **Ito-to-Stratonovich correction** introduces an additional drift term.

---
This proof should be **uploaded to your GitHub** repository as per the assignment instructions.