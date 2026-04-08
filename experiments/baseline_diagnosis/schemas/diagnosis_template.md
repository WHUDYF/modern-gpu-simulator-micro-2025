# Diagnosis Report: {workload} [{experiment_id}]

**Date:** {date}
**Hardware:** {hardware}
**Input features:** {features_path}
**Mechanisms enabled:** {mechanisms_list}
**Diagnoser:** {manual|skill}

---

## Stage A: Software Utilization Check

### Utilization metrics

| Kernel | waves_per_sm | achieved_occupancy | grid_size |
|--------|-------------|---------------------|-----------|
| ... | ... | ... | ... |

### Stage A Verdict

- [ ] Workload utilizes hardware adequately (waves_per_sm ≥ 4, occupancy ≥ 80%)
- [ ] If no, prescribe Class A fix first (software configuration change)

### Class A Prescription (if needed)

[Only filled if Stage A verdict is negative]

---

## Stage B: Architecture Bottleneck Analysis

### Per-kernel bottleneck identification

For each kernel:
- **Dominant bottleneck (highest distance-to-roof utilization):**
- **Secondary bottleneck:**
- **Cross-source reasoning (trace + hw_stats):**

### Mechanism-informed insights (only if mechanisms enabled)

- **Squash findings:** behavior phases identified, phase transitions
- **Batch findings:** clusters, outlier units
- **Delta findings:** hot/cold fields, field correlations

---

## Class B Prescriptions

### Prescription B.{N}: {title}

**Target kernel:** {kernel_name}

**Modification:**
```
<parameter name> <old value> → <new value>
```

**Reason:**
[Evidence trail, which features support this]

**Expected effect:**
- Metric X: direction ± magnitude
- Metric Y: direction ± magnitude

**Expected cost:**
[Area/power/other tradeoff]

**Verification:**
- Modify: [config file path]
- Rerun: [command]
- Compare: [which metric]
- Success criterion: [threshold]

**Confidence:** HIGH / MEDIUM / LOW

**Control kernel (unchanged prediction):** [kernel that should NOT change if prescription is correctly scoped]

---

## Summary

- Total prescriptions: N
- High confidence: M
- Prescriptions that use mechanism features: K (only applies when mechanisms enabled)
- Prescriptions that would not exist without mechanism features: J (only applies when mechanisms enabled)
