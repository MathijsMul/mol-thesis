from fol_gen_complex import axioms, filter_axioms, interpret

#s = 'v	( ( two ( not Romans ) ) ( ( not hate ) ( all ( not Germans ) ) ) )	( ( ( not some ) Germans ) ( like ( all ( not Romans ) ) ) )'

da_subj1 = ''
da_subj2 = 'not'
va_subj1 = 'not'
va_subj2 = ''
da_obj1 = ''
da_obj2 = ''
det_obj1 = 'all'
det_obj2 = 'all'
va_obj1 = 'not'
va_obj2 = 'not'
det_subj1 = 'two'
det_subj2 = 'some'
na_subj1 = 'not'
na_subj2 = ''
n_subj1 = 'Romans'
n_subj2 = 'Germans'
v_subj1 = 'hate'
v_subj2 = 'like'
n_obj1 = 'Germans'
n_obj2 = 'Romans'

s = [[[(da_subj1, da_subj2), (det_subj1, det_subj2)], [(na_subj1, na_subj2), (n_subj1, n_subj2)]], \
                 [[(va_subj1, va_subj2), (v_subj1, v_subj2)], [[(da_obj1, da_obj2), (det_obj1, det_obj2)], [(va_obj1, va_obj2), (n_obj1, n_obj2)]]]]


filtered_axioms = filter_axioms(axioms, det_subj1, det_subj2, na_subj1, na_subj2, \
                                            n_subj1, n_subj2, v_subj1, v_subj2, n_obj1, n_obj2)


#rel = interpret(s, filtered_axioms)
#print('============================== Prover9 ===============================\nProver9 (64) version 2009-11A, November 2009.\nProcess 6732 was started by Mathijs on vpn-stud-146-50-148-100.vpn.uva.nl,\nSat Nov 11 18:18:56 2017\nThe command was "/usr/local/bin/prover9".\n============================== end of head ===========================\n\n============================== INPUT =================================\nassign(max_seconds,1).\nclear(auto_denials).\n\nformulas(assumptions).\n-(all x (Romans(x) | Germans(x))).\n-(all x all y (hate(x,y) | like(x,y))).\n(all x -(Romans(x) & Germans(x))).\n(all x all y -(hate(x,y) & like(x,y))).\n-(exists x exists y (-Romans(x) & -Romans(y) & x != y & -(all v (-Germans(v) -> hate(x,v) & hate(y,v))))).\nend_of_list.\n\nformulas(goals).\n-(exists x (Germans(x) & (all y (-Romans(y) -> like(x,y))))).\nend_of_list.\n\n============================== end of input ==========================\n\n============================== PROCESS NON-CLAUSAL FORMULAS ==========\n\n% Formulas that are not ordinary clauses:\n1 -(all x (Romans(x) | Germans(x))) # label(non_clause).  [assumption].\n2 -(all x all y (hate(x,y) | like(x,y))) # label(non_clause).  [assumption].\n3 (all x -(Romans(x) & Germans(x))) # label(non_clause).  [assumption].\n4 (all x all y -(hate(x,y) & like(x,y))) # label(non_clause).  [assumption].\n5 -(exists x exists y (-Romans(x) & -Romans(y) & x != y & -(all v (-Germans(v) -> hate(x,v) & hate(y,v))))) # label(non_clause).  [assumption].\n6 -(exists x (Germans(x) & (all y (-Romans(y) -> like(x,y))))) # label(non_clause) # label(goal).  [goal].\n\n============================== end of process non-clausal formulas ===\n\n============================== PROCESS INITIAL CLAUSES ===============\n\n% Clauses before input processing:\n\nformulas(usable).\nend_of_list.\n\nformulas(sos).\n-Romans(c1).  [clausify(1)].\n-Germans(c1).  [clausify(1)].\n-hate(c2,c3).  [clausify(2)].\n-like(c2,c3).  [clausify(2)].\n-Romans(x) | -Germans(x).  [clausify(3)].\n-hate(x,y) | -like(x,y).  [clausify(4)].\nRomans(x) | Romans(y) | y = x | Germans(z) | hate(x,z).  [clausify(5)].\nRomans(x) | Romans(y) | y = x | Germans(z) | hate(y,z).  [clausify(5)].\nGermans(c4).  [deny(6)].\nRomans(x) | like(c4,x).  [deny(6)].\nend_of_list.\n\nformulas(demodulators).\nend_of_list.\n\n============================== PREDICATE ELIMINATION =================\n\nNo predicates eliminated.\n\n============================== end predicate elimination =============\n\nTerm ordering decisions:\nPredicate symbol precedence:  predicate_order([ =, Romans, Germans, hate, like ]).\nFunction symbol precedence:  function_order([ c1, c2, c3, c4 ]).\nAfter inverse_order:  (no changes).\nUnfolding symbols: (none).\n\nAuto_inference settings:\n  % set(paramodulation).  % (positive equality literals)\n  % set(binary_resolution).  % (non-Horn)\n  % set(neg_ur_resolution).  % (non-Horn, less than 100 clauses)\n\nAuto_process settings:\n  % set(factor).  % (non-Horn)\n  % set(unit_deletion).  % (non-Horn)\n\nkept:      7 -Romans(c1).  [clausify(1)].\nkept:      8 -Germans(c1).  [clausify(1)].\nkept:      9 -hate(c2,c3).  [clausify(2)].\nkept:      10 -like(c2,c3).  [clausify(2)].\nkept:      11 -Romans(x) | -Germans(x).  [clausify(3)].\nkept:      12 -hate(x,y) | -like(x,y).  [clausify(4)].\nkept:      13 Romans(x) | Romans(y) | y = x | Germans(z) | hate(x,z).  [clausify(5)].\nkept:      14 Romans(x) | Romans(y) | y = x | Germans(z) | hate(y,z).  [clausify(5)].\nkept:      15 Germans(c4).  [deny(6)].\nkept:      16 Romans(x) | like(c4,x).  [deny(6)].\n\n============================== end of process initial clauses ========\n\n============================== CLAUSES FOR SEARCH ====================\n\n% Clauses after input processing:\n\nformulas(usable).\nend_of_list.\n\nformulas(sos).\n7 -Romans(c1).  [clausify(1)].\n8 -Germans(c1).  [clausify(1)].\n9 -hate(c2,c3).  [clausify(2)].\n10 -like(c2,c3).  [clausify(2)].\n11 -Romans(x) | -Germans(x).  [clausify(3)].\n12 -hate(x,y) | -like(x,y).  [clausify(4)].\n13 Romans(x) | Romans(y) | y = x | Germans(z) | hate(x,z).  [clausify(5)].\n14 Romans(x) | Romans(y) | y = x | Germans(z) | hate(y,z).  [clausify(5)].\n15 Germans(c4).  [deny(6)].\n16 Romans(x) | like(c4,x).  [deny(6)].\nend_of_list.\n\nformulas(demodulators).\nend_of_list.\n\n============================== end of clauses for search =============\n\n============================== SEARCH ================================\n\n% Starting search at 0.01 seconds.\n\ngiven #1 (I,wt=2): 7 -Romans(c1).  [clausify(1)].\n\ngiven #2 (I,wt=2): 8 -Germans(c1).  [clausify(1)].\n\ngiven #3 (I,wt=3): 9 -hate(c2,c3).  [clausify(2)].\n\ngiven #4 (I,wt=3): 10 -like(c2,c3).  [clausify(2)].\n\ngiven #5 (I,wt=4): 11 -Romans(x) | -Germans(x).  [clausify(3)].\n\ngiven #6 (I,wt=6): 12 -hate(x,y) | -like(x,y).  [clausify(4)].\n\ngiven #7 (I,wt=12): 13 Romans(x) | Romans(y) | y = x | Germans(z) | hate(x,z).  [clausify(5)].\n\ngiven #8 (I,wt=12): 14 Romans(x) | Romans(y) | y = x | Germans(z) | hate(y,z).  [clausify(5)].\n\ngiven #9 (I,wt=2): 15 Germans(c4).  [deny(6)].\n\ngiven #10 (I,wt=5): 16 Romans(x) | like(c4,x).  [deny(6)].\n\ngiven #11 (A,wt=9): 17 Romans(c2) | Romans(x) | c2 = x | Germans(c3).  [resolve(13,e,9,a),flip(c)].\n\ngiven #12 (F,wt=2): 18 -Romans(c4).  [resolve(15,a,11,b)].\n\ngiven #13 (T,wt=5): 19 Romans(x) | -hate(c4,x).  [resolve(16,b,12,b)].\n\ngiven #14 (T,wt=7): 24 Romans(x) | c4 = x | Germans(x).  [factor(21,a,b)].\n\x07-------- Proof 1 -------- \n\n============================== PROOF =================================\n\n% Proof 1 at 0.01 (+ 0.00) seconds.\n% Length of proof is 18.\n% Level of proof is 6.\n% Maximum clause weight is 12.000.\n% Given clauses 14.\n\n1 -(all x (Romans(x) | Germans(x))) # label(non_clause).  [assumption].\n3 (all x -(Romans(x) & Germans(x))) # label(non_clause).  [assumption].\n4 (all x all y -(hate(x,y) & like(x,y))) # label(non_clause).  [assumption].\n5 -(exists x exists y (-Romans(x) & -Romans(y) & x != y & -(all v (-Germans(v) -> hate(x,v) & hate(y,v))))) # label(non_clause).  [assumption].\n6 -(exists x (Germans(x) & (all y (-Romans(y) -> like(x,y))))) # label(non_clause) # label(goal).  [goal].\n7 -Romans(c1).  [clausify(1)].\n8 -Germans(c1).  [clausify(1)].\n11 -Romans(x) | -Germans(x).  [clausify(3)].\n12 -hate(x,y) | -like(x,y).  [clausify(4)].\n14 Romans(x) | Romans(y) | y = x | Germans(z) | hate(y,z).  [clausify(5)].\n15 Germans(c4).  [deny(6)].\n16 Romans(x) | like(c4,x).  [deny(6)].\n18 -Romans(c4).  [resolve(15,a,11,b)].\n19 Romans(x) | -hate(c4,x).  [resolve(16,b,12,b)].\n21 Romans(x) | Romans(y) | c4 = y | Germans(x).  [resolve(19,b,14,e),unit_del(c,18)].\n24 Romans(x) | c4 = x | Germans(x).  [factor(21,a,b)].\n25 c4 = c1.  [resolve(24,c,8,a),unit_del(a,7)].\n31 $F.  [back_rewrite(15),rewrite([25(1)]),unit_del(a,8)].\n\n============================== end of proof ==========================\n\n============================== STATISTICS ============================\n\nGiven=14. Generated=35. Kept=24. proofs=1.\nUsable=9. Sos=1. Demods=1. Limbo=6, Disabled=18. Hints=0.\nKept_by_rule=0, Deleted_by_rule=0.\nForward_subsumed=10. Back_subsumed=0.\nSos_limit_deleted=0. Sos_displaced=0. Sos_removed=0.\nNew_demodulators=1 (0 lex), Back_demodulated=8. Back_unit_deleted=0.\nDemod_attempts=34. Demod_rewrites=9.\nRes_instance_prunes=0. Para_instance_prunes=0. Basic_paramod_prunes=0.\nNonunit_fsub_feature_tests=4. Nonunit_bsub_feature_tests=13.\nMegabytes=0.07.\nUser_CPU=0.01, System_CPU=0.00, Wall_clock=0.\n\n============================== end of statistics =====================\n\n============================== end of search =========================\n\nTHEOREM PROVED\n\nTHEOREM PROVED\n\nExiting with 1 proof.\n\n------ process 6732 exit (max_proofs) ------\n\x07\nProcess 6732 exit (max_proofs) Sat Nov 11 18:18:56 2017\n' )

s = 'v	( ( two ( not Romans ) ) ( ( not hate ) ( all ( not Germans ) ) ) )	( ( ( not some ) Germans ) ( like ( all ( not Romans ) ) ) )'
print(s.split())

#
# with open('bulk2dets4negs_6bulk.txt', 'r') as f:
#     for idx, l in enumerate(f):
#         continue
#     print(idx)


