function dmrg_target(PH, psi0::MPS; reverse_step = false, target_eigenvalue, kwargs...)
    function dmrg_target_solver(PH, t, psi0; kwargs...)
        H = contract(PH, ITensor(1.0))
        D, U = eigen(H; ishermitian = true)
        u = uniqueind(U, H)
        max_overlap, max_ind = findmin(dⱼ -> abs(dⱼ - target_eigenvalue), diag(D))
        U_max = U * dag(onehot(u => max_ind))
        return U_max, nothing
    end
    t = Inf
    return tdvp(dmrg_target_solver, PH, t, psi0; reverse_step, kwargs...)
end

# function dmrg_target(PH, psi0::MPS; reverse_step=false, target_eigenvalue, linsolve_kwargs, eigsolve_kwargs, kwargs...)
#   function dmrg_target_solver(PH, t, psi0; kwargs...)
#     u = eigsolve_target(PH, target_eigenvalue, psi0; linsolve_kwargs, eigsolve_kwargs)
#     return u, nothing
#   end
#   t = Inf
#   return tdvp(dmrg_target_solver, PH, t, psi0; reverse_step, kwargs...)
# end
