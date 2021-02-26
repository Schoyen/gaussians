from gaussians import G2D, OD2D


g_00 = G2D((0, 0), 1, (1, 2))
# g_01 = G2D((0, 1), 0.3, (0, 0))
g_21 = G2D((2, 1), 0.7, (-0.3, 0))

od_00_00 = OD2D(g_00, g_00)
od_00_21 = OD2D(g_00, g_21)
od_21_00 = OD2D(g_21, g_00)
od_21_21 = OD2D(g_21, g_21)


def print_expansion(od):
    print(od.G_a)
    print(od.G_b)
    for t in range(3):
        for u in range(3):
            print(f"\t({t}, {u}) = {od.E(t, u)}")


print_expansion(od_00_00)
print_expansion(od_00_21)
print_expansion(od_21_00)
print_expansion(od_21_21)
