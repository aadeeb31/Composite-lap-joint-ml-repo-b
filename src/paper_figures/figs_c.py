from figlib import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
import joblib
from figs_b import CLS_C


def _A(name): return pickle.load(open(f'{ANA}/{name}.pkl', 'rb'))
def _t(name): return [t for n, _, t, _ in DS if n == name][0]


def consensus_summary():
    C = {}; FAMRHO = {}
    for n, *_ in DS:
        A = _A(n); PI = A['PI'].clip(lower=0); N = PI / PI.sum(); sc = A['scores'].set_index('Model'); key = 'Test_f1' if A['clf'] else 'Test_r2'
        good = [m for m in N.columns if sc.loc[m, key] > 0.5]; C[n] = N[good].mean(1)
        rho = pd.DataFrame(np.clip(spearmanr(N.values)[0], 0, 1), index=N.columns, columns=N.columns); fam = pd.Series({m: S.MODEL_FAMILY[m] for m in N.columns}); FAMRHO[n] = rho, fam
    C = pd.DataFrame(C); tops = C.mean(1).sort_values(ascending=False).index[:16][::-1]
    fig = plt.figure(figsize=(15.4, 7.6)); gs = GridSpec(1, 2, width_ratios=[1.0, 0.95], wspace=0.18); a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a1.grid(False)
    for j, n in enumerate(C.columns):
        for i, f in enumerate(tops):
            v = C.loc[f, n]; a0.scatter(j, i, s=30 + 3600 * v, fc=S.GROUPC[S.group_of(f)], ec='k', lw=1.3, alpha=.92, zorder=3)
            if v > .06: a0.text(j, i, f'{v*100:.0f}', ha='center', va='center', fontsize=10.4, fontweight='bold', color='white', zorder=4)
    a0.set_xticks(range(len(C.columns))); a0.set_xticklabels([SHORT[n].replace('Composite-composite','Composite-\ncomposite').replace('CFRP-CFRP','CFRP-\nCFRP').replace('\nmax load','\nload').replace('\nfailure mode','\nfailure mode') for n in C.columns], fontsize=11.5); a0.set_yticks(range(len(tops))); a0.set_yticklabels([S.nice(f) for f in tops], fontsize=13.0); a0.set_xlim(-.6, len(C.columns) - .4); a0.set_ylim(-.7, len(tops) - .3)
    a0.set_title('Consensus importance (% of model total)', fontsize=15.0, pad=10); S.panel(a0, 'a', dx=-0.02)
    M = np.zeros((8, 8)); cnt = np.zeros((8, 8))
    for n, (rho, fam) in FAMRHO.items():
        for a in rho.index:
            for b in rho.columns:
                if a == b: continue
                i, j = FIDX[fam[a]], FIDX[fam[b]]; M[i, j] += rho.loc[a, b]; cnt[i, j] += 1
    M = np.where(cnt > 0, M / np.maximum(cnt, 1), np.nan)
    a1.pcolormesh(np.arange(9) - .5, np.arange(9) - .5, M, cmap=S.SEQ, vmin=0.2, vmax=1, edgecolors='white', linewidth=1.5)
    for i in range(8):
        for j in range(8):
            if not np.isnan(M[i, j]): a1.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=11.2, fontweight='bold', color='white' if M[i, j] > .62 else 'k')
    a1.set_xticks(range(8)); a1.set_xticklabels(S.FAM, rotation=90, ha='center', fontsize=12.7); a1.set_yticks(range(8)); a1.set_yticklabels([f.replace('Nearest neighbour','Nearest\nneighbour') for f in S.FAM], fontsize=12.7); a1.invert_yaxis(); a1.set_aspect('equal'); a1.tick_params(length=0)
    for i, f in enumerate(S.FAM): a1.add_patch(Rectangle((i - .5, -.5 - .75), 1, .55, fc=S.FAMC[f], ec='k', lw=1, clip_on=False))
    a1.set_title('Family-to-family agreement: mean Spearman ρ\n(importance rankings, averaged over the ' + str(len(FAMRHO)) + ' tasks)', fontsize=15.0, pad=62); S.panel(a1, 'b', dx=-0.05, dy=1.2)
    h = [Patch(fc=c, ec='k', label=g) for g, c in S.GROUPC.items()]; fig.legend(handles=h, loc='lower center', ncol=6, fontsize=12.3, bbox_to_anchor=(0.3, -0.08), title='Feature group', title_fontsize=12.3)
    save(fig, 'fig10_consensus_what_matters_where'); return C, M


def pdp_overlay(name):
    A = _A(name)
    if 'pdp' not in A: return
    P = A['pdp']; unit = UNIT[name]; fig, axs = plt.subplots(2, 2, figsize=(12.6, 9.4)); axs = axs.ravel(); fig.subplots_adjust(hspace=0.3, wspace=0.22)
    for ax, f in zip(axs, P['top']):
        dd = P['data'][f]; gr = dd['grid']; allc = np.array([c for _, c in dd['curves'].values()])
        ax.fill_between(gr, allc.min(0), allc.max(0), color='#E6E6E6', ec='none', zorder=1, label='range across families')
        for fam in S.FAM:
            if fam not in dd['curves']: continue
            mn, cv = dd['curves'][fam]; ax.plot(gr, cv, color=S.FAMC[fam], lw=2.6, zorder=3, path_effects=None); ax.plot(gr, cv, ls='', marker=S.FAMM[fam], ms=7.5, mfc=S.FAMC[fam], mec='k', mew=1.0, markevery=4, zorder=4)
        if gr.min() > 0 and gr.max() / gr.min() >= 15: ax.set_xscale('log'); ax.set_xticks([t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if gr.min() * 0.95 <= t <= gr.max() * 1.05]); ax.set_xticklabels([f'{t:g}' for t in ax.get_xticks()]); ax.minorticks_off()
        yl = ax.get_ylim(); ax.vlines(dd['rug'], yl[0], yl[0] + 0.035 * (yl[1] - yl[0]), color='k', lw=1.2, zorder=5); ax.set_ylim(yl)
        ax.set_xlabel(S.nice(f) + (' (log scale)' if ax.get_xscale() == 'log' else '')); ax.set_ylabel(f'Partial dependence ({unit})'); S.panel(ax, 'abcd'[list(axs).index(ax)], dx=-0.12)
    h = [Line2D([], [], color=S.FAMC[f], marker=S.FAMM[f], mfc=S.FAMC[f], mec='k', lw=2.6, ms=8.5, label=f'{f} ({A["pdp"]["reps"][f]})') for f in S.FAM if f in A['pdp']['reps']] + [Patch(fc='#E6E6E6', ec='none', label='range across families')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=12.5, bbox_to_anchor=(0.5, -0.045), title='Best model of each family;  rug = data deciles', title_fontsize=13.0)
    fig.suptitle(f'{_t(name)}: the same feature, seen by every family', fontweight='bold', fontsize=17.6, y=0.955)
    save(fig, f'fig11_pdp_family_overlay_{name}')


def clf_calibration(name):
    A = _A(name)
    if 'rel' not in A: return
    REL = A['rel']; sc = A['scores'].set_index('Model'); reps = sc.loc[list(REL)].groupby('Family').Test_f1.idxmax()
    fig = plt.figure(figsize=(16, 5.9)); gs = GridSpec(1, 3, wspace=0.28, width_ratios=[1, 1, 1]); a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2])
    a0.plot([0, 1], [0, 1], 'k', lw=2)
    for fam, m in reps.items():
        r = REL[m]; bx = np.array(r['bx'], float); by = np.array(r['by'], float); ok = ~np.isnan(bx); a0.plot(bx[ok], by[ok], color=S.FAMC[fam], lw=2.2, zorder=2)
        a0.scatter(bx[ok], by[ok], s=20 + 8 * np.array(r['bn'])[ok], marker=S.FAMM[fam], fc=S.FAMC[fam], ec='k', lw=1.0, zorder=3)
    a0.set_xlabel('Predicted confidence (max class probability)'); a0.set_ylabel('Observed accuracy'); a0.set_xlim(.1, 1.03); a0.set_ylim(0, 1.03); a0.set_title('Reliability diagram (marker area ~ samples in bin)', fontsize=14.3); S.panel(a0, 'a', dx=-0.13)
    e = pd.Series({m: REL[m]['ece'] for m in REL}).sort_values()
    for k, (m, v) in enumerate(e.items()):
        f = S.MODEL_FAMILY[m]; a1.barh(k, v, color=S.FAMC[f], ec='k', lw=1.3, height=.72); a1.text(v + .004, k, f'{v:.3f}', va='center', fontsize=11.4, fontweight='bold')
    a1.set_yticks(range(len(e))); a1.set_yticklabels(e.index, fontsize=12.5, fontweight='bold'); a1.invert_yaxis(); a1.set_xlabel('Expected calibration error (lower = better)'); a1.set_title('Calibration error by model', fontsize=14.3); S.panel(a1, 'b', dx=-0.18)
    for fam, m in reps.items():
        r = np.array(REL[m]['retention']); x = np.arange(1, len(r) + 1) / len(r); a2.plot(x, r, color=S.FAMC[fam], lw=2.4, label=f'{fam} ({m})'); a2.plot(x[::max(len(x) // 6, 1)], r[::max(len(x) // 6, 1)], ls='', marker=S.FAMM[fam], ms=8, mfc=S.FAMC[fam], mec='k')
    a2.set_xlabel('Fraction of test samples answered (most confident first)'); a2.set_ylabel('Accuracy on answered samples'); a2.set_title('Selective prediction: abstain when unsure', fontsize=14.3); a2.set_xlim(0, 1.02); a2.set_ylim(.4, 1.02); a2.legend(fontsize=10.7, loc='lower left'); S.panel(a2, 'c', dx=-0.13)
    fig.suptitle(f'{_t(name)}: can the class probabilities be trusted?', fontweight='bold', fontsize=17.6, y=1.02)
    save(fig, f'fig13_calibration_{name}')


def confusion_f1(name):
    from sklearn.metrics import confusion_matrix, f1_score
    A = _A(name); cn = class_names(); r = RES[RES.File == name]; best = r.sort_values('Score').iloc[-1]; P = PRED[(name, best.Model)]
    classes = sorted(set(P['yte']) | set(P['pte'])); labs = [cn.get(c, str(c)) for c in classes]
    cm = confusion_matrix(P['yte'], P['pte'], labels=classes); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = plt.figure(figsize=(16, 7.0)); gs = GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.55); a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a0.grid(False); a1.grid(False)
    a0.pcolormesh(np.arange(len(classes) + 1) - .5, np.arange(len(classes) + 1) - .5, cmn, cmap=S.CONF, vmin=0, vmax=1, edgecolors='k', linewidth=1.2)
    for i in range(len(classes)):
        for j in range(len(classes)): a0.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=14.3, fontweight='bold', color='white' if cmn[i, j] > .5 else 'k')
    a0.set_xticks(range(len(classes))); a0.set_xticklabels(labs, rotation=0, fontsize=13.7); a0.set_yticks(range(len(classes))); a0.set_yticklabels([f'{l}  (n={cm[i].sum()})' for i, l in enumerate(labs)], fontsize=12.5); a0.invert_yaxis(); a0.set_aspect('equal')
    a0.set_xlabel('Predicted'); a0.set_ylabel('Observed'); a0.set_title(f'{best.Model}: counts (colour = row share)', fontsize=15.0); S.panel(a0, 'a', dx=-0.3)
    mods = list(ordered_models(r).Model); F = np.array([f1_score(PRED[(name, m)]['yte'], PRED[(name, m)]['pte'], labels=classes, average=None, zero_division=0) for m in mods])
    a1.pcolormesh(np.arange(len(classes) + 1) - .5, np.arange(len(mods) + 1) - .5, F, cmap=S.SCORE, vmin=0, vmax=1, edgecolors='white', linewidth=1.4)
    for i in range(len(mods)):
        for j in range(len(classes)): a1.text(j, i, f'{F[i, j]:.2f}', ha='center', va='center', fontsize=11.2, fontweight='bold', color='white' if F[i, j] < .25 else 'k')
    a1.set_xticks(range(len(classes))); a1.set_xticklabels(labs, rotation=0, fontsize=13.7); a1.set_yticks(range(len(mods))); a1.set_yticklabels(mods, fontsize=12.2, fontweight='bold'); a1.invert_yaxis(); a1.tick_params(length=0)
    a1.tick_params(axis='y', pad=16)
    for i, m in enumerate(mods): a1.add_patch(Rectangle((-.5 - .13, i - .5), .13, 1, fc=S.FAMC[S.MODEL_FAMILY[m]], ec='k', lw=1, clip_on=False))
    a1.set_title('Per-class F1 of every model (rows grouped by family)', fontsize=15.0); S.panel(a1, 'b', dx=-0.2)
    fig.suptitle(_t(name), fontweight='bold', fontsize=17.6, y=1.0)
    save(fig, f'fig14_confusion_and_class_f1_{name}')


if __name__ == '__main__':
    import sys
    for n in [a for a in sys.argv[1:]] or [d[0] for d in DS]:
        if not os.path.exists(f'{ANA}/{n}.pkl'): continue
        if n.endswith('MaxLoad') or n.endswith('LSS'): pdp_overlay(n)
        else: clf_calibration(n); confusion_f1(n)
