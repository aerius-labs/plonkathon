from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis


@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    setup: Setup
    program: Program
    pk: CommonPreprocessedInput

    def __init__(self, setup: Setup, program: Program):
        self.group_order = program.group_order
        self.setup = setup
        self.program = program
        self.pk = program.common_preprocessed_input()

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1)

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2)

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        setup = self.setup
        group_order = self.group_order

        if None not in witness:
            witness[None] = 0

        # Compute wire assignments for A, B, C, corresponding:
        # - A_values: witness[program.wires()[i].L]
        # - B_values: witness[program.wires()[i].R]
        # - C_values: witness[program.wires()[i].O]
        A_values = [Scalar(witness[wire.L]) for wire in program.wires()]
        B_values = [Scalar(witness[wire.R]) for wire in program.wires()]
        C_values = [Scalar(witness[wire.O]) for wire in program.wires()]

        while(len(A_values) < group_order):
            A_values.append(Scalar(0))
        while(len(B_values) < group_order):
            B_values.append(Scalar(0))
        while(len(C_values) < group_order):
            C_values.append(Scalar(0))

        # Construct A, B, C Lagrange interpolation polynomials for
        # A_values, B_values, C_values
        self.A = Polynomial(A_values, Basis.LAGRANGE)
        self.B = Polynomial(B_values, Basis.LAGRANGE)
        self.C = Polynomial(C_values, Basis.LAGRANGE)

        # Compute a_1, b_1, c_1 commitments to A, B, C polynomials
        a_1 = setup.commit(self.A)
        b_1 = setup.commit(self.B)
        c_1 = setup.commit(self.C)

        # Sanity check that witness fulfils gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        )

        # Return a_1, b_1, c_1
        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order
        setup = self.setup

        # Using A, B, C, values, and pk.S1, pk.S2, pk.S3, compute
        # Z_values for permutation grand product polynomial Z
        #
        # Note the convenience function:
        #       self.rlc(val1, val2) = val_1 + self.beta * val_2 + gamma
        Z_values = [Scalar(1)]
        roots_of_unity = Scalar.roots_of_unity(group_order)
        for i in range(group_order):
            Z_values.append(
                Z_values[-1] *
                self.rlc(self.A.values[i], roots_of_unity[i]) *
                self.rlc(self.B.values[i], 2 * roots_of_unity
                [i]) *
                self.rlc(self.C.values[i], 3 * roots_of_unity
                [i]) /
                self.rlc(self.A.values[i], self.pk.S1.values[i]) /
                self.rlc(self.B.values[i], self.pk.S2.values[i]) /
                self.rlc(self.C.values[i], self.pk.S3.values[i])
            )

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0

        # Construct Z, Lagrange interpolation polynomial for Z_values
        self.Z = Polynomial(Z_values, Basis.LAGRANGE)
        # Cpmpute z_1 commitment to Z polynomial
        z_1 = setup.commit(self.Z)

        # Return z_1
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order
        setup = self.setup

        # Compute the quotient polynomial

        # List of roots of unity at 4x fineness, i.e. the powers of µ
        # where µ^(4n) = 1
        roots4x = Polynomial(Scalar.roots_of_unity(group_order * 4), Basis.LAGRANGE)
        self.roots4x = roots4x

        # Using self.fft_expand, move A, B, C into coset extended Lagrange basis
        expanded_A = self.fft_expand(self.A)
        expanded_B = self.fft_expand(self.B)
        expanded_C = self.fft_expand(self.C)

        self.expanded_A = expanded_A
        self.expanded_B = expanded_B
        self.expanded_C = expanded_C
        
        # Expand public inputs polynomial PI into coset extended Lagrange
        expanded_PI = self.fft_expand(self.PI)

        # Expand selector polynomials pk.QL, pk.QR, pk.QM, pk.QO, pk.QC
        # into the coset extended Lagrange basis
        expanded_QL = self.fft_expand(self.pk.QL)
        expanded_QR = self.fft_expand(self.pk.QR)
        expanded_QM = self.fft_expand(self.pk.QM)
        expanded_QO = self.fft_expand(self.pk.QO)
        expanded_QC = self.fft_expand(self.pk.QC)

        self.expanded_QL = expanded_QL
        self.expanded_QR = expanded_QR
        self.expanded_QM = expanded_QM
        self.expanded_QO = expanded_QO
        self.expanded_QC = expanded_QC

        # Expand permutation grand product polynomial Z into coset extended
        # Lagrange basis
        expanded_Z = self.fft_expand(self.Z)
        self.expanded_Z = expanded_Z

        # Expand shifted Z(ω) into coset extended Lagrange basis
        Z_shift = expanded_Z.shift(4)

        # Expand permutation polynomials pk.S1, pk.S2, pk.S3 into coset
        # extended Lagrange basis
        expanded_S1 = self.fft_expand(self.pk.S1)
        expanded_S2 = self.fft_expand(self.pk.S2)
        expanded_S3 = self.fft_expand(self.pk.S3)

        self.expanded_S1 = expanded_S1
        self.expanded_S2 = expanded_S2
        self.expanded_S3 = expanded_S3

        # Compute Z_H = X^N - 1, also in evaluation form in the coset
        self.Z_H = Polynomial([(Scalar(root) * self.fft_cofactor) ** group_order - 1
                          for root in roots4x.values], Basis.LAGRANGE)

        # Compute L0, the Lagrange basis polynomial that evaluates to 1 at x = 1 = ω^0
        # and 0 at other roots of unity
        self.L0 = Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        # Expand L0 into the coset extended Lagrange basis
        L0_big = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )

        # Compute the quotient polynomial (called T(x) in the paper)
        # It is only possible to construct this polynomial if the following
        # equations are true at all roots of unity {1, w ... w^(n-1)}:
        # 1. All gates are correct:
        #    A * QL + B * QR + A * B * QM + C * QO + PI + QC = 0
        #
        # 2. The permutation accumulator is valid:
        #    Z(wx) = Z(x) * (rlc of A, X, 1) * (rlc of B, 2X, 1) *
        #                   (rlc of C, 3X, 1) / (rlc of A, S1, 1) /
        #                   (rlc of B, S2, 1) / (rlc of C, S3, 1)
        #    rlc = random linear combination: term_1 + beta * term2 + gamma * term3
        #
        # 3. The permutation accumulator equals 1 at the start point
        #    (Z - 1) * L0 = 0
        #    L0 = Lagrange polynomial, equal at all roots of unity except 1

        gate_check = (
            expanded_QL * expanded_A +
            expanded_QR * expanded_B +
            expanded_QM * expanded_A * expanded_B +
            expanded_QO * expanded_C +
            expanded_PI +
            expanded_QC
        )

        p_accum = ((
            self.rlc(expanded_A, roots4x * self.fft_cofactor) *
            self.rlc(expanded_B, roots4x * (self.fft_cofactor * 2)) *
            self.rlc(expanded_C, roots4x * (self.fft_cofactor * 3))
            ) * expanded_Z
            - (
                self.rlc(expanded_A, expanded_S1) *
                self.rlc(expanded_B, expanded_S2) *
                self.rlc(expanded_C, expanded_S3)
                ) * Z_shift
            )
        
        L0_check = (expanded_Z - Scalar(1)) * L0_big

        QUOT_big = (gate_check + 
                    p_accum * self.alpha + 
                    L0_check * self.alpha**2) / self.Z_H
        
        # Sanity check: QUOT has degree < 3n
        assert (
            self.expanded_evals_to_coeffs(QUOT_big).values[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Split up T into T1, T2 and T3 (needed because T has degree 3n - 4, so is
        # too big for the trusted setup)
        coeff = self.expanded_evals_to_coeffs(QUOT_big).values
        T1 = Polynomial(coeff[:group_order], Basis.MONOMIAL).fft()
        T2 = Polynomial(coeff[group_order: 2*group_order], Basis.MONOMIAL).fft()
        T3 = Polynomial(coeff[2*group_order: 3*group_order], Basis.MONOMIAL).fft()

        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

        fft_cofactor = self.fft_cofactor
        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            T1.barycentric_eval(fft_cofactor)
            + T2.barycentric_eval(fft_cofactor) * fft_cofactor**group_order
            + T3.barycentric_eval(fft_cofactor) * fft_cofactor ** (group_order * 2)
        ) == QUOT_big.values[0]

        print("Generated T1, T2, T3 polynomials")

        # Compute commitments t_lo_1, t_mid_1, t_hi_1 to T1, T2, T3 polynomials
        t_lo_1, t_mid_1, t_hi_1 = setup.commit(T1), setup.commit(T2), setup.commit(T3)

        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:
        # Compute evaluations to be used in constructing the linearization polynomial.

        # Compute a_eval = A(zeta)
        # Compute b_eval = B(zeta)
        # Compute c_eval = C(zeta)
        # Compute s1_eval = pk.S1(zeta)
        # Compute s2_eval = pk.S2(zeta)
        # Compute z_shifted_eval = Z(zeta * ω)
        self.a_eval = self.A.barycentric_eval(self.zeta)
        self.b_eval = self.B.barycentric_eval(self.zeta)
        self.c_eval = self.C.barycentric_eval(self.zeta)
        self.s1_eval = self.pk.S1.barycentric_eval(self.zeta)
        self.s2_eval = self.pk.S2.barycentric_eval(self.zeta)
        self.s3_eval = self.pk.S3.barycentric_eval(self.zeta)
        self.z_shifted_eval = self.Z.barycentric_eval(self.zeta * Scalar.root_of_unity(self.group_order))

        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval)

    def round_5(self) -> Message5:
        # Evaluate the Lagrange basis polynomial L0 at zeta
        L0_eval = self.L0.barycentric_eval(self.zeta)

        # Evaluate the vanishing polynomial Z_H(X) = X^n - 1 at zeta
        Z_H_eval = self.zeta ** self.group_order - 1

        # Move T1, T2, T3 into the coset extended Lagrange basis
        expanded_T1 = self.fft_expand(self.T1)
        expanded_T2 = self.fft_expand(self.T2)
        expanded_T3 = self.fft_expand(self.T3)

        # Move pk.QL, pk.QR, pk.QM, pk.QO, pk.QC into the coset extended Lagrange basis
        # Move Z into the coset extended Lagrange basis
        # Move pk.S3 into the coset extended Lagrange basis
        expanded_S3 = self.fft_expand(self.pk.S3)
        c_eval = Polynomial([self.c_eval] * self.group_order * 4, Basis.LAGRANGE)

        # Compute the "linearization polynomial" R. This is a clever way to avoid
        # needing to provide evaluations of _all_ the polynomials that we are
        # checking an equation betweeen: instead, we can "skip" the first
        # multiplicand in each term. The idea is that we construct a
        # polynomial which is constructed to equal 0 at Z only if the equations
        # that we are checking are correct, and which the verifier can reconstruct
        # the KZG commitment to, and we provide proofs to verify that it actually
        # equals 0 at Z
        #
        zerotimesalpha = (
              self.expanded_QM * self.a_eval * self.b_eval
            + self.expanded_QL * self.a_eval
            + self.expanded_QR * self.b_eval
            + self.expanded_QO * c_eval
            + self.PI.barycentric_eval(self.zeta)
            + self.expanded_QC
        )

        onetimesalpha = (
              self.expanded_Z
            * self.rlc(self.a_eval, self.zeta)
            * self.rlc(self.b_eval, self.zeta * 2)
            * self.rlc(c_eval, self.zeta * 3)
            -
              self.rlc(c_eval, expanded_S3)
            * self.rlc(self.a_eval, self.s1_eval)
            * self.rlc(self.b_eval, self.s2_eval)
            * self.z_shifted_eval
        )

        twotimesalpha = (
            (self.expanded_Z - Scalar(1)) * L0_eval
        )

        expanded_R = (
            zerotimesalpha
            + onetimesalpha * self.alpha
            + twotimesalpha * self.alpha ** 2
            - (  expanded_T1
               + expanded_T2 * self.zeta ** self.group_order
               + expanded_T3 * self.zeta ** (self.group_order * 2)
               ) * Z_H_eval
        )

        # In order for the verifier to be able to reconstruct the commitment to R,
        # it has to be "linear" in the proof items, hence why we can only use each
        # proof item once; any further multiplicands in each term need to be
        # replaced with their evaluations at Z, which do still need to be provided
        R_coeffs = self.expanded_evals_to_coeffs(expanded_R).values
        R = Polynomial(R_coeffs[:self.group_order], Basis.MONOMIAL).fft()

        # Commit to R
        r_1 = self.setup.commit(R)
        # Sanity-check R
        assert R.barycentric_eval(self.zeta) == 0

        print("Generated linearization polynomial R")

        # Generate proof that W(z) = 0 and that the provided evaluations of
        # A, B, C, S1, S2 are correct

        # Move A, B, C into the coset extended Lagrange basis
        # Move pk.S1, pk.S2 into the coset extended Lagrange basis

        # In the COSET EXTENDED LAGRANGE BASIS,
        # Construct W_Z = (
        #     R
        #   + v * (A - a_eval)
        #   + v**2 * (B - b_eval)
        #   + v**3 * (C - c_eval)
        #   + v**4 * (S1 - s1_eval)
        #   + v**5 * (S2 - s2_eval)
        # ) / (X - zeta)
        expanded_W_z = (
            expanded_R
          + (self.expanded_A - self.a_eval) * self.v   
          + (self.expanded_B - self.b_eval) * self.v**2
          + (self.expanded_C - self.c_eval) * self.v**3
          + (self.expanded_S1 - self.s1_eval) * self.v**4
          + (self.expanded_S2 - self.s2_eval) * self.v**5
        ) / (self.roots4x * self.fft_cofactor - self.zeta)

        # Check that degree of W_z is not greater than n
        W_z_coeffs = self.expanded_evals_to_coeffs(expanded_W_z).values
        assert W_z_coeffs[self.group_order:] == [0] * (self.group_order * 3)

        # Compute W_z_1 commitment to W_z
        W_z = Polynomial(W_z_coeffs[:self.group_order], Basis.MONOMIAL).fft()
        W_z_1 = self.setup.commit(W_z)

        # Generate proof that the provided evaluation of Z(z*w) is correct. This
        # awkwardly different term is needed because the permutation accumulator
        # polynomial Z is the one place where we have to check between adjacent
        # coordinates, and not just within one coordinate.
        # In other words: Compute W_zw = (Z - z_shifted_eval) / (X - zeta * ω)
        expanded_W_zw = ((self.expanded_Z - self.z_shifted_eval)
                / (self.roots4x * self.fft_cofactor
                   - Scalar.root_of_unity(self.group_order) * self.zeta))
        
        W_zw_coeffs = self.expanded_evals_to_coeffs(expanded_W_zw).values
        W_zw = Polynomial(W_zw_coeffs[:self.group_order], Basis.MONOMIAL).fft()
        # Check that degree of W_z is not greater than n
        assert W_zw_coeffs[self.group_order:] == [0] * (self.group_order * 3)

        # Compute W_zw_1 commitment to W_zw
        W_zw_1 = self.setup.commit(W_zw)
        
        print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma
