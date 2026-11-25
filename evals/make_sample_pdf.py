"""Creates evals/data/os_notes.pdf: a small, known document for the golden dataset."""
from pathlib import Path

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = {
    "Processes and Threads": [
        "A process is a program in execution. It has its own address space, program counter, "
        "registers and open files. The operating system tracks each process with a Process "
        "Control Block (PCB), which stores its state, priority and scheduling information.",
        "A process moves through five states: new, ready, running, waiting and terminated. "
        "A context switch saves the state of the running process in its PCB and loads the state "
        "of the next process. Context switches are pure overhead because no useful work is done.",
        "A thread is the smallest unit of CPU execution. Threads of the same process share code, "
        "data and open files but have their own stack and registers. Threads are cheaper to create "
        "and switch than processes.",
    ],
    "CPU Scheduling": [
        "CPU scheduling decides which ready process runs next. Goals include high CPU utilization, "
        "high throughput, and low turnaround, waiting and response times.",
        "First-Come, First-Served (FCFS) runs processes in arrival order and can cause the convoy "
        "effect, where short jobs wait behind a long one. Shortest Job First (SJF) gives the minimum "
        "average waiting time but needs to know burst lengths in advance and can starve long jobs.",
        "Round Robin (RR) gives each process a fixed time quantum. If the quantum is too large, RR "
        "behaves like FCFS; if it is too small, context-switch overhead dominates. Priority "
        "scheduling can cause starvation, which is solved by aging: slowly raising the priority of "
        "waiting processes.",
    ],
    "Memory Management": [
        "Paging divides physical memory into fixed-size frames and logical memory into pages of the "
        "same size. A page table maps page numbers to frame numbers. Paging removes external "
        "fragmentation but can cause internal fragmentation in the last page.",
        "A Translation Lookaside Buffer (TLB) is a small, fast cache of recent page-table entries "
        "that speeds up address translation. Virtual memory lets a process run even when only part "
        "of it is in memory; a page fault occurs when a needed page is not in memory.",
        "Page replacement algorithms choose which page to evict. FIFO is simple but suffers from "
        "Belady's anomaly, where more frames can cause more page faults. LRU replaces the least "
        "recently used page. Thrashing happens when a system spends more time paging than executing.",
    ],
    "Deadlocks": [
        "A deadlock is a situation where a set of processes are each waiting for a resource held by "
        "another process in the set, so none can proceed.",
        "Four conditions must all hold for a deadlock: mutual exclusion, hold and wait, no "
        "preemption, and circular wait. Preventing any one of them prevents deadlock.",
        "The Banker's algorithm avoids deadlock by granting a request only if the system stays in a "
        "safe state, meaning there is some order in which every process can finish. Deadlocks can "
        "also be detected with a resource-allocation graph and recovered from by terminating "
        "processes or preempting resources.",
    ],
}


def main(out: Path = Path(__file__).parent / "data" / "os_notes.pdf"):
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES.items()):
        if i:
            story.append(PageBreak())
        story += [Paragraph(f"Operating Systems Notes - {title}", styles["Heading1"]), Spacer(1, 8)]
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 6)]
    SimpleDocTemplate(str(out), title="Operating Systems Notes").build(story)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
