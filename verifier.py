import py_ecc.bn128 as b
from utils import *
from dataclasses import dataclass
from curve import *
from transcript import Transcript
from poly import Polynomial, Basis


@dataclass
class VerificationKey:
    """Verification key"""

    # we set this to some power of 2 (so that we can FFT over it), that is at least the number of constraints we have (so we can Lagrange interpolate them)
    group_order: int
    # [q_M(x)]₁ (commitment to multiplication selector polynomial)
    Qm: G1Point
    # [q_L(x)]₁ (commitment to left selector polynomial)
    Ql: G1Point
    # [q_R(x)]₁ (commitment to right selector polynomial)
    Qr: G1Point
    # [q_O(x)]₁ (commitment to output selector polynomial)
    Qo: G1Point
    # [q_C(x)]₁ (commitment to constants selector polynomial)
    Qc: G1Point
    # [S_σ1(x)]₁ (commitment to the first permutation polynomial S_σ1(X))
    S1: G1Point
    # [S_σ2(x)]₁ (commitment to the second permutation polynomial S_σ2(X))
    S2: G1Point
    # [S_σ3(x)]₁ (commitment to the third permutation polynomial S_σ3(X))
    S3: G1Point
    # [x]₂ = xH, where H is a generator of G_2
    X_2: G2Point
    # nth root of unity (i.e. ω^1), where n is the program's group order.
    w: Scalar

    # More optimized version that tries hard to minimize pairings and
    # elliptic curve multiplications, but at the cost of being harder
    # to understand and mixing together a lot of the computations to
    # efficiently batch them
    def verify_proof(self, group_order: int, pf, public=[]) -> bool:
        # 4. Compute challenges
        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf)
        self.beta, self.gamma = beta, gamma
        
        # 5. Compute zero polynomial evaluation Z_H(ζ) = ζ^n - 1
        Z_H_eval = zeta ** group_order - 1

        # 6. Compute Lagrange polynomial evaluation L_0(ζ)
        

        # 7. Compute public input polynomial evaluation PI(ζ).

        # Compute the constant term of R. This is not literally the degree-0
        # term of the R polynomial; rather, it's the portion of R that can
        # be computed directly, without resorting to elliptic cutve commitments

        # Compute D = (R - r0) + u * Z, and E and F

        # Run one pairing check to verify the last two checks.
        # What's going on here is a clever re-arrangement of terms to check
        # the same equations that are being checked in the basic version,
        # but in a way that minimizes the number of EC muls and even
        # compressed the two pairings into one. The 2 pairings -> 1 pairing
        # trick is basically to replace checking
        #
        # Y1 = A * (X - a) and Y2 = B * (X - b)
        #
        # with
        #
        # Y1 + A * a = A * X
        # Y2 + B * b = B * X
        #
        # so at this point we can take a random linear combination of the two
        # checks, and verify it with only one pairing.

        return False

    # Basic, easier-to-understand version of what's going on
    def verify_proof_unoptimized(self, group_order: int, pf, public=[]) -> bool:
        # 4. Compute challenges
        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf)
        self.beta, self.gamma = beta, gamma
        pf = pf.flatten()

        # 5. Compute zero polynomial evaluation Z_H(ζ) = ζ^n - 1
        Z_H_eval = zeta ** group_order - 1

        # 6. Compute Lagrange polynomial evaluation L_0(ζ)
        L_0_eval = Z_H_eval / (group_order * (zeta - 1))

        # 7. Compute public input polynomial evaluation PI(ζ).
        PI = Polynomial(
            [Scalar(-v) for v in public]
            + [Scalar(0) for _ in range(self.group_order - len(public))],
            Basis.LAGRANGE,
        )
        # PI_eval = PI.barycentric_eval(zeta)

        # Recover the commitment to the linearization polynomial R,
        # exactly the same as what was created by the prover
        zerotimesalpha = ec_lincomb([
              (self.Qm, pf["a_eval"] * pf["b_eval"]),
              (self.Ql, pf["a_eval"]),
              (self.Qr, pf["b_eval"]),
              (self.Qo, pf["c_eval"]),
              (b.G1, PI.barycentric_eval(zeta)),
              (self.Qc, Scalar(1)),
        ])
        
        onetimesalpha = ec_lincomb([
             (pf["z_1"],
              self.rlc(pf["a_eval"], zeta)
            * self.rlc(pf["b_eval"], zeta * 2)
            * self.rlc(pf["c_eval"], zeta * 3)
             ),

             (self.S3,
              -beta
            * self.rlc(pf["a_eval"], pf["s1_eval"])
            * self.rlc(pf["b_eval"], pf["s2_eval"])
            * pf["z_shifted_eval"]
             ),

             (b.G1,
              -(pf["c_eval"] + gamma)
            * self.rlc(pf["a_eval"], pf["s1_eval"])
            * self.rlc(pf["b_eval"], pf["s2_eval"])
            * pf["z_shifted_eval"]
              )
        ])

        twotimesalpha = ec_lincomb([
             (pf["z_1"], L_0_eval),
             (b.G1, -L_0_eval)
        ])

        T_sum = ec_lincomb([
             (pf["t_lo_1"], 1),
             (pf["t_mid_1"], zeta ** self.group_order),
             (pf["t_hi_1"], zeta ** (self.group_order * 2))
        ])

        r1 = ec_lincomb([
             (zerotimesalpha, 1),
             (onetimesalpha, alpha),
             (twotimesalpha, alpha**2),
             (T_sum, -Z_H_eval)
        ])

        # Verify that R(z) = 0 and the prover-provided evaluations
        # A(z), B(z), C(z), S1(z), S2(z) are all correct
        lhs = b.pairing(
             ec_lincomb([
                  (self.X_2, 1),
                  (b.G2, -zeta)
             ]),
             pf["W_z_1"]
        )

        rhs = b.pairing(b.G2, 
            ec_lincomb(
            [
                (r1, 1),
                (pf["a_1"], v),
                (b.G1, -pf["a_eval"] * v),
                (pf["b_1"], v**2),
                (b.G1, -pf["b_eval"] * v**2),
                (pf["c_1"], v**3),
                (b.G1, -pf["c_eval"] * v**3),
                (self.S1, v**4),
                (b.G1, -pf["s1_eval"] * v**4),
                (self.S2, v**5),
                (b.G1, -pf["s2_eval"] * v**5),
            ]
            ))
        
        assert lhs == rhs

        # Verify that the provided value of Z(zeta*w) is correct
        lhs = b.pairing(
             ec_lincomb([
                  (self.X_2, 1),
                  (b.G2, -zeta * Scalar.root_of_unity(group_order))
             ]),
             pf["W_zw_1"]
        )

        rhs = b.pairing(b.G2,
             ec_lincomb([
                  (pf["z_1"], 1),
                  (b.G1, -pf["z_shifted_eval"])
             ])
        )

        assert lhs == rhs
        
        return True

    # Compute challenges (should be same as those computed by prover)
    def compute_challenges(
        self, proof
    ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
        transcript = Transcript(b"plonk")
        beta, gamma = transcript.round_1(proof.msg_1)
        alpha, _fft_cofactor = transcript.round_2(proof.msg_2)
        zeta = transcript.round_3(proof.msg_3)
        v = transcript.round_4(proof.msg_4)
        u = transcript.round_5(proof.msg_5)

        return beta, gamma, alpha, zeta, v, u

    def rlc(self, term_1, term_2):
            return term_1 + term_2 * self.beta + self.gamma