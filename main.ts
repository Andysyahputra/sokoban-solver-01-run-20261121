// Sokoban solver: room decomposition + push-optimal A* per room.
//
// One run answers a set of 5 instances. Each level is a hallway with rooms
// off it; a box pushed out of a room can never come back, so rooms are
// independent sub-problems and the level optimum is the sum of the rooms'.
// Each room is solved push-optimally (cost = pushes) with an A* search:
//   - state: sorted box cells + player cell, packed into one number
//   - heuristic: minimum-cost matching of boxes to goals over BFS distances
//   - pruning: static dead cells (pull-reachability from goals), frozen
//     boxes, 2x2 box blocks, and boxes that can reach no goal
// Rooms are stitched together with deterministic hallway walks.
// Deterministic: fixed direction orders, work budgets (no clock, no random).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const ROOM_CAP_1 = 250000;
const ROOM_CAP_2 = 3000000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

function solveLevel(text: string): string {
  const rows = text.split('\n').filter((r) => r.length > 0);
  if (rows.length === 0) return '';
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const n = w * h;
  const wall = new Uint8Array(n);
  const goals: number[] = [];
  let boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') wall[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  const goalSet = new Set(goals);
  const boxSet = new Set(boxes);
  const at = (x: number, y: number): number => y * w + x;
  const inB = (i: number): boolean => i >= 0 && i < n;

  // ---- rooms: connected components of floor cells off the hallway row ----
  const hallwayRow = Math.floor(player / w);
  const seen = new Uint8Array(n);
  const rooms: { cells: number[]; cellSet: Set<number>; boxes: number[]; goals: number[]; door: number; entry: number }[] = [];
  for (let s = 0; s < n; s++) {
    if (wall[s] || seen[s] || Math.floor(s / w) === hallwayRow) continue;
    const comp: number[] = [];
    const stack = [s];
    seen[s] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      comp.push(c);
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q = at(nx, ny);
        if (wall[q] || seen[q] || ny === hallwayRow) continue;
        seen[q] = 1;
        stack.push(q);
      }
    }
    const rBoxes = comp.filter((c) => boxSet.has(c)).sort((a, b) => a - b);
    if (rBoxes.length === 0) continue;
    const rGoals = comp.filter((c) => goalSet.has(c)).sort((a, b) => a - b);
    // door: the component cell next to a free hallway cell; entry: that cell.
    let door = -1;
    let entry = -1;
    const sorted = comp.slice().sort((a, b) => a - b);
    for (const c of sorted) {
      const x = c % w;
      const y = Math.floor(c / w);
      for (const dy of [-1, 1]) {
        const ny = y + dy;
        if (ny !== hallwayRow) continue;
        const e = at(x, ny);
        if (!wall[e]) {
          door = c;
          entry = e;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) continue;
    rooms.push({ cells: comp, cellSet: new Set(comp), boxes: rBoxes, goals: rGoals, door, entry });
  }
  const covered = new Set<number>();
  for (const r of rooms) for (const b of r.boxes) covered.add(b);
  if (rooms.length === 0 || boxes.some((b) => !covered.has(b))) return '';

  // ---- per-room push-optimal A* ----
  rooms.sort((a, b) => a.door - b.door);
  const roomPlans: { box: number; dir: number }[][] = [];
  for (const room of rooms) {
    let plan = solveRoom(room, ROOM_CAP_1);
    if (plan === null) plan = solveRoom(room, ROOM_CAP_2);
    if (plan === null) return '';
    roomPlans.push(plan);
  }

  // ---- stitch: walks + pushes into one move string ----
  let moves = '';
  let cur = player;
  const allBoxes = boxes.slice();
  const walk = (from: number, to: number): string | null => {
    if (from === to) return '';
    const prev = new Int32Array(n).fill(-1);
    const prevCh = new Array<string>(n).fill('');
    const q = [from];
    prev[from] = from;
    let head = 0;
    while (head < q.length) {
      const c = q[head++]!;
      if (c === to) break;
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q2 = at(nx, ny);
        if (wall[q2] || prev[q2] >= 0 || allBoxes.includes(q2)) continue;
        prev[q2] = c;
        prevCh[q2] = CH[d]!;
        q.push(q2);
      }
    }
    if (prev[to] < 0) return null;
    let s = '';
    for (let c = to; c !== from; c = prev[c]!) s = prevCh[c]! + s;
    return s;
  };

  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const plan = roomPlans[ri]!;
    const toEntry = walk(cur, room.entry);
    if (toEntry === null) return '';
    moves += toEntry;
    cur = room.entry;
    for (const p of plan) {
      const stand = p.box - (DY[p.dir]! * w + DX[p.dir]!);
      const wpath = walk(cur, stand);
      if (wpath === null) return '';
      moves += wpath;
      moves += CH[p.dir]!;
      const bi = allBoxes.indexOf(p.box);
      allBoxes[bi] = p.box + (DY[p.dir]! * w + DX[p.dir]!);
      allBoxes.sort((a, b) => a - b);
      cur = p.box;
    }
    const exit = walk(cur, room.entry);
    if (exit === null) return '';
    moves += exit;
    cur = room.entry;
  }

  // ---- safety: replay the moves; only keep a fully legal solution ----
  if (!replay(text, moves)) {
    process.stderr.write(`sokoban: replay failed, discarding solution\n`);
    return '';
  }
  return moves;

  // ---- room solver (push A*) ----
  function solveRoom(
    room: { cells: number[]; cellSet: Set<number>; boxes: number[]; goals: number[]; door: number; entry: number },
    cap: number,
  ): { box: number; dir: number }[] | null {
    const floor = new Set<number>(room.cells);
    floor.add(room.entry);
    const nb = room.boxes.length;
    const ng = room.goals.length;
    if (nb === 0 || nb !== ng) return null;

    // BFS distances from each goal over room cells only (box travel).
    const distFromGoal: Int16Array[] = room.goals.map((g) => {
      const dist = new Int16Array(n).fill(-1);
      const q = [g];
      dist[g] = 0;
      let head = 0;
      while (head < q.length) {
        const c = q[head++]!;
        const x = c % w;
        const y = Math.floor(c / w);
        for (let d = 0; d < 4; d++) {
          const nx = x + DX[d]!;
          const ny = y + DY[d]!;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q2 = at(nx, ny);
          if (!room.cellSet.has(q2) || dist[q2] >= 0) continue;
          dist[q2] = dist[c]! + 1;
          q.push(q2);
        }
      }
      return dist;
    });

    // Static dead cells: pull-reachability from goals. A box position is
    // reachable if it can be pulled there from a goal; the player may stand
    // on any floor cell (room cells + the entry cell) for a pull.
    const reach = new Uint8Array(n);
    const rq = room.goals.slice();
    for (const g of rq) reach[g] = 1;
    let rhead = 0;
    while (rhead < rq.length) {
      const c = rq[rhead++]!;
      const x = c % w;
      const y = Math.floor(c / w);
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d]!;
        const ny = y + DY[d]!;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const q2 = at(nx, ny);
        if (!room.cellSet.has(q2) || reach[q2]) continue;
        // Pulling the box from c to q2: the player stands beyond q2,
        // at 2*q2 - c, and pulls the box toward themselves.
        const sx = x + 2 * DX[d]!;
        const sy = y + 2 * DY[d]!;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const s = at(sx, sy);
        if (!floor.has(s)) continue;
        reach[q2] = 1;
        rq.push(q2);
      }
    }
    const dead = new Uint8Array(n);
    for (const c of room.cells) if (!reach[c]) dead[c] = 1;
    for (const b of room.boxes) if (dead[b]) return null;

    // Minimum-matching heuristic, by bitmask-DP assignment (O(n * 2^n),
    // exact, and far cheaper than enumerating goal permutations when a room
    // has more boxes). Memoised on the sorted box configuration, since the
    // same configuration is reached from many player positions.
    const hMemo = new Map<string, number>();
    const heuristic = (bs: number[]): number => {
      const key = bs.join(",");
      const hit = hMemo.get(key);
      if (hit !== undefined) return hit;
      const INF = 1e9;
      const dp = new Array<number>(1 << nb).fill(INF);
      dp[0] = 0;
      for (let mask = 0; mask < (1 << nb); mask++) {
        const cur = dp[mask]!;
        if (cur >= INF) continue;
        // Number of boxes already assigned = popcount(mask).
        let i = 0;
        let m = mask;
        while (m) {
          i += m & 1;
          m >>= 1;
        }
        if (i >= nb) continue;
        for (let j = 0; j < ng; j++) {
          if (mask & (1 << j)) continue;
          const d = distFromGoal[j]![bs[i]!]!;
          if (d < 0) continue;
          const nm = mask | (1 << j);
          const v = cur + d;
          if (v < dp[nm]!) dp[nm] = v;
        }
      }
      const best = dp[(1 << nb) - 1]!;
      const result = best >= INF ? Infinity : best;
      hMemo.set(key, result);
      return result;
    };

    const packBoxes = (bs: number[]): number => {
      let k = 255;
      for (let i = nb - 1; i >= 0; i--) k = (k << 8) | bs[i]!;
      return k >>> 0;
    };
    const unpackBoxes = (k: number): number[] => {
      const bs: number[] = [];
      for (let i = 0; i < nb; i++) bs.push((k >> (8 * i)) & 255);
      return bs;
    };
    const stateKey = (p: number, bp: number): number => p + bp * 256;

    const startBoxes = room.boxes.slice();
    const startBp = packBoxes(startBoxes);
    const h0 = heuristic(startBoxes);
    if (h0 === Infinity) return null;

    // A* state arrays.
    const sp: number[] = [room.entry];
    const sb: number[] = [startBp];
    const sg: number[] = [0];
    const sparent: number[] = [-1];
    const spushBox: number[] = [-1];
    const spushDir: number[] = [-1];
    const best = new Map<number, number>();
    best.set(stateKey(room.entry, startBp), 0);
    // Binary heap of [f, h, seq, stateIdx]; ties: smaller h, then earlier seq.
    const heap: [number, number, number, number][] = [];
    let seq = 0;
    const hless = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
      a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
    const hpush = (e: [number, number, number, number]): void => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!hless(heap[i]!, heap[p]!)) break;
        [heap[i], heap[p]] = [heap[p]!, heap[i]!];
        i = p;
      }
    };
    const hpop = (): [number, number, number, number] => {
      const top = heap[0]!;
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= heap.length) break;
          if (c + 1 < heap.length && hless(heap[c + 1]!, heap[c]!)) c++;
          if (!hless(heap[c]!, heap[i]!)) break;
          [heap[i], heap[c]] = [heap[c]!, heap[i]!];
          i = c;
        }
      }
      return top;
    };
    hpush([h0, h0, seq++, 0]);

    const boxOcc = new Uint8Array(n);
    let expanded = 0;
    while (heap.length > 0) {
      const [, , , s] = hpop();
      const p = sp[s]!;
      const bp = sb[s]!;
      const g = sg[s]!;
      const key = stateKey(p, bp);
      if (best.get(key) !== s) continue;
      const bs = unpackBoxes(bp);
      if (bs.every((b) => goalSet.has(b))) {
        // Player must be able to walk out through the door.
        if (canReach(p, room.door, bs)) {
          const plan: { box: number; dir: number }[] = [];
          for (let t = s; t > 0; t = sparent[t]!) {
            plan.push({ box: spushBox[t]!, dir: spushDir[t]! });
          }
          plan.reverse();
          return plan;
        }
      }
      if (expanded >= cap) return null;
      expanded++;
      // Player reachability over floor cells minus boxes.
      for (const b of bs) boxOcc[b] = 1;
      const reachP = new Uint8Array(n);
      const pq = [p];
      reachP[p] = 1;
      let phead = 0;
      while (phead < pq.length) {
        const c = pq[phead++]!;
        const x = c % w;
        const y = Math.floor(c / w);
        for (let d = 0; d < 4; d++) {
          const nx = x + DX[d]!;
          const ny = y + DY[d]!;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q2 = at(nx, ny);
          if (!floor.has(q2) || boxOcc[q2] || reachP[q2]) continue;
          reachP[q2] = 1;
          pq.push(q2);
        }
      }
      for (let bi = 0; bi < nb; bi++) {
        const b = bs[bi]!;
        const bx = b % w;
        const by = Math.floor(b / w);
        for (let d = 0; d < 4; d++) {
          const dx = bx + DX[d]!;
          const dy = by + DY[d]!;
          if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
          const dest = at(dx, dy);
          if (!room.cellSet.has(dest) || boxOcc[dest] || dead[dest]) continue;
          const sx = bx - DX[d]!;
          const sy = by - DY[d]!;
          if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
          const stand = at(sx, sy);
          if (!reachP[stand]) continue;
          const nb2 = bs.slice();
          nb2[bi] = dest;
          nb2.sort((a, b) => a - b);
          if (!deadOk(nb2)) continue;
          const nbp = packBoxes(nb2);
          const nkey = stateKey(b, nbp);
          const seenG = best.get(nkey);
          if (seenG !== undefined && sg[seenG]! <= g + 1) continue;
          const hh = heuristic(nb2);
          if (hh === Infinity) continue;
          const t = sp.length;
          sp.push(b);
          sb.push(nbp);
          sg.push(g + 1);
          sparent.push(s);
          spushBox.push(b);
          spushDir.push(d);
          best.set(nkey, t);
          hpush([g + 1 + hh, hh, seq++, t]);
        }
      }
      for (const b of bs) boxOcc[b] = 0;
    }
    return null;

    function deadOk(bs: number[]): boolean {
      // Frozen boxes, by fixpoint: a box is frozen when every push direction
      // is blocked by a wall or by an already-frozen box. A box blocked only
      // by a movable neighbour is merely stuck for now, not dead.
      const occ = new Set(bs);
      const frozen = new Set<number>();
      let changed = true;
      while (changed) {
        changed = false;
        for (const b of bs) {
          if (frozen.has(b)) continue;
          const bx = b % w;
          const by = Math.floor(b / w);
          let movable = false;
          for (let d = 0; d < 4; d++) {
            const dx = bx + DX[d]!;
            const dy = by + DY[d]!;
            const sx = bx - DX[d]!;
            const sy = by - DY[d]!;
            if (dx < 0 || dy < 0 || dx >= w || dy >= h) continue;
            if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
            const dest = at(dx, dy);
            const stand = at(sx, sy);
            if (!room.cellSet.has(dest) || frozen.has(dest)) continue;
            if (!floor.has(stand) || frozen.has(stand)) continue;
            movable = true;
            break;
          }
          if (!movable) {
            frozen.add(b);
            changed = true;
          }
        }
      }
      for (const b of frozen) if (!goalSet.has(b)) return false;
      // 2x2 block of boxes with one off a goal: no box in it can ever move.
      for (const b of bs) {
        if (goalSet.has(b)) continue;
        const bx = b % w;
        const by = Math.floor(b / w);
        for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
          const x0 = bx + ox;
          const y0 = by + oy;
          if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
          const sq = [at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)];
          if (sq.every((c) => occ.has(c)) && sq.some((c) => !goalSet.has(c))) return false;
        }
      }
      return true;
    }

    function canReach(from: number, to: number, bs: number[]): boolean {
      if (from === to) return true;
      const occ = new Set(bs);
      const vis = new Uint8Array(n);
      const q = [from];
      vis[from] = 1;
      let head = 0;
      while (head < q.length) {
        const c = q[head++]!;
        const x = c % w;
        const y = Math.floor(c / w);
        for (let d = 0; d < 4; d++) {
          const nx = x + DX[d]!;
          const ny = y + DY[d]!;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q2 = at(nx, ny);
          if (!floor.has(q2) || occ.has(q2) || vis[q2]) continue;
          if (q2 === to) return true;
          vis[q2] = 1;
          q.push(q2);
        }
      }
      return false;
    }
  }

  // ---- replay a move string on the level; true iff legal and solved ----
  function replay(text: string, moves: string): boolean {
    const rows = text.split('\n').filter((r) => r.length > 0);
    const grid: string[][] = rows.map((r) => r.split(''));
    let px = 0;
    let py = 0;
    for (let y = 0; y < grid.length; y++) {
      for (let x = 0; x < grid[y]!.length; x++) {
        if (grid[y]![x] === '@' || grid[y]![x] === '+') {
          px = x;
          py = y;
        }
      }
    }
    const cell = (x: number, y: number): string => grid[y]?.[x] ?? '#';
    for (const ch of moves) {
      if (ch === '\n' || ch === ' ' || ch === '\r' || ch === '\t') continue;
      const d = 'udlr'.indexOf(ch.toLowerCase());
      if (d < 0) return false;
      const nx = px + DX[d]!;
      const ny = py + DY[d]!;
      const c = cell(nx, ny);
      if (c === '#') return false;
      if (c === '$' || c === '*') {
        const bx = nx + DX[d]!;
        const by = ny + DY[d]!;
        const bc = cell(bx, by);
        if (bc === '#' || bc === '$' || bc === '*') return false;
        grid[ny]![nx] = c === '*' ? '.' : ' ';
        grid[by]![bx] = bc === '.' ? '*' : '$';
      }
      grid[py]![px] = grid[py]![px] === '+' ? '.' : ' ';
      px = nx;
      py = ny;
      grid[py]![px] = cell(px, py) === '.' || cell(px, py) === '*' ? '+' : '@';
    }
    for (const row of grid) {
      for (const c of row) if (c === '$') return false;
    }
    return true;
  }
}

for (const id of readdirSync(IN).sort()) {
  const moves = solveLevel(readFileSync(join(IN, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), `${moves}\n`);
}
