import math

class SurfaceElo:
    def __init__(self):
        # schema: { surface: { player_id: {"matches": int, "elo": float} } }
        self.ratings = {}

    def _ensure_player(self, player_id, surface):
        if surface not in self.ratings:
            self.ratings[surface] = {}
        if player_id not in self.ratings[surface]:
            self.ratings[surface][player_id] = {"matches": 0, "elo": 1500.0}

    def get_k_factor(self, matches):
        return 32.0 if matches < 30 else 20.0

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id, loser_id, surface):
        self._ensure_player(winner_id, surface)
        self._ensure_player(loser_id, surface)

        w_state = self.ratings[surface][winner_id]
        l_state = self.ratings[surface][loser_id]

        w_rating = w_state["elo"]
        l_rating = l_state["elo"]

        e_w = self.expected_score(w_rating, l_rating)
        e_l = self.expected_score(l_rating, w_rating)

        k_w = self.get_k_factor(w_state["matches"])
        k_l = self.get_k_factor(l_state["matches"])

        # Update
        w_state["elo"] += k_w * (1.0 - e_w)
        l_state["elo"] += k_l * (0.0 - e_l)

        w_state["matches"] += 1
        l_state["matches"] += 1


class Glicko2:
    def __init__(self, tau=0.5):
        self.tau = tau
        # schema: { player_id: {"mu": 0.0, "phi": 2.0, "sigma": 0.06} } 
        # (normalized scale: rating = (mu * 173.7178) + 1500)
        self.players = {}

    def _ensure_player(self, player_id):
        if player_id not in self.players:
            self.players[player_id] = {"mu": 0.0, "phi": 2.0, "sigma": 0.06}

    def g(self, phi):
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / (math.pi**2))

    def E(self, mu, mu_j, phi_j):
        return 1.0 / (1.0 + math.exp(-self.g(phi_j) * (mu - mu_j)))

    def update(self, winner_id, loser_id):
        self._ensure_player(winner_id)
        self._ensure_player(loser_id)
        
        w_state = self.players[winner_id].copy()
        l_state = self.players[loser_id].copy()
        
        self._calculate_new_rating(winner_id, w_state, [(l_state["mu"], l_state["phi"], 1.0)])
        self._calculate_new_rating(loser_id, l_state, [(w_state["mu"], w_state["phi"], 0.0)])

    def _calculate_new_rating(self, p_id, p_state, results):
        mu = p_state["mu"]
        phi = p_state["phi"]
        sigma = p_state["sigma"]

        v_inv = 0.0
        for mu_j, phi_j, s_j in results:
            g_j = self.g(phi_j)
            E_j = self.E(mu, mu_j, phi_j)
            v_inv += (g_j**2) * E_j * (1.0 - E_j)
        
        if v_inv == 0: v_inv = 1e-6
        v = 1.0 / v_inv

        delta_sum = 0.0
        for mu_j, phi_j, s_j in results:
            g_j = self.g(phi_j)
            E_j = self.E(mu, mu_j, phi_j)
            delta_sum += g_j * (s_j - E_j)
        delta = v * delta_sum

        a = math.log(sigma**2)
        def f(x):
            num = math.exp(x) * (delta**2 - phi**2 - v - math.exp(x))
            den = 2.0 * (phi**2 + v + math.exp(x))**2
            return num / den - (x - a) / (self.tau**2)

        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA = f(A)
        fB = f(B)

        while abs(B - A) > 0.000001:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2.0
            B = C
            fB = fC
            
        sigma_new = math.exp(A / 2.0)
        phi_star = math.sqrt(phi**2 + sigma_new**2)
        phi_new = 1.0 / math.sqrt(1.0 / (phi_star**2) + 1.0 / v)
        mu_new = mu + phi_new**2 * delta_sum

        self.players[p_id]["mu"] = mu_new
        self.players[p_id]["phi"] = phi_new
        self.players[p_id]["sigma"] = sigma_new

