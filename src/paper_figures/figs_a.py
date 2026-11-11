from figlib import *
from matplotlib.colors import Normalize


def fig04_arena():
    fig, axs = plt.subplots(3, 3, figsize=(22, 22.1), gridspec_kw=dict(height_ratios=[1, 1, 0.62])); axs = axs.ravel(); fig.subplots_adjust(hspace=0.22, wspace=0.42)
    for ax, (n, rel, ttl, kind) in zip(axs, DS):
        r = RES[RES.File == n]; y = 0; yt = []; yl = []; cols = []
        for f in S.FAM:
            q = r[r.Family == f].sort_values('Score', ascending=False)
            if q.empty: continue
            for _, z in q.iterrows():
                ax.scatter(z.Score, y, s=90 + 2600 * z.Test5seeds_sd, marker=S.FAMM[f], fc=S.FAMC[f], ec='k', lw=1.3, zorder=3)
                ax.errorbar(z.Score, y, xerr=z.Test5seeds_sd, color='k', lw=1.3, capsize=3, zorder=2)
                yt.append(y); yl.append(z.Model); cols.append(S.FAMC[f]); y += 1
            ax.plot([q.Score.mean()] * 2, [y - len(q) - .35, y - .65], color=S.FAMC[f], lw=3.2, zorder=1, solid_capstyle='butt')
            y += 0.7
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontweight='bold', fontsize=12.5); ax.invert_yaxis(); ax.set_ylim(y - .5, -.9)
        for t, c in zip(ax.get_yticklabels(), cols): t.set_color(c if c not in ('#F59E0B', '#7BC62D') else '#222')
        ax.set_title(SHORT[n].replace('\n', ' · '), fontsize=16); ax.set_xlabel('R² (random test)' if kind == 'r2' else 'macro-F1 (random test)')
        rng = r.Score.max() - r.Score.min(); ax.set_xlim(r.Score.min() - 0.06 * rng - 0.02, min(1.03, r.Score.max() + 0.06 * rng + 0.02)); S.panel(ax, 'abcdef'[[d[0] for d in DS].index(n)], dx=-0.30)
    axs[6].axis('off'); axs[8].axis('off'); ax = axs[7]; ax.axis('off'); family_legend(axs[6], loc='center', ncol=2, title='Model family')
    for lab, v, yy in [('0.00', 0.00, 0.52), ('0.02', 0.02, 0.40), ('0.05', 0.05, 0.27)]:
        ax.scatter([0.25], [yy], s=90 + 2600 * v, fc='#DDD', ec='k', lw=1.4, transform=ax.transAxes); ax.text(0.38, yy, f'5-seed s.d. = {lab}', transform=ax.transAxes, va='center', fontsize=15)
    ax.text(0.5, 0.62, 'Marker size and error bar = s.d. of the score over five random splits', ha='center', va='center', transform=ax.transAxes, fontsize=13.5, style='italic', wrap=True)
    ax.text(0.5, 0.12, 'Coloured bar = family mean.', ha='center', va='center', transform=ax.transAxes, fontsize=14, style='italic')
    save(fig, 'fig04_family_arena')


if __name__ == '__main__':
    for f in (fig01_leaderboard, fig02_gap, fig03_bubble, fig04_arena): f()
