use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

fn compress_line_left(line: [u8; 4]) -> ([u8; 4], u32, bool) {
    let nonzero: Vec<u8> = line.into_iter().filter(|&x| x != 0).collect();
    let mut out = Vec::with_capacity(4);
    let mut reward: u32 = 0;
    let mut index = 0;

    while index < nonzero.len() {
        if index + 1 < nonzero.len() && nonzero[index] == nonzero[index + 1] {
            let merged = nonzero[index] + 1;
            out.push(merged);
            reward += 1u32 << merged;
            index += 2;
        } else {
            out.push(nonzero[index]);
            index += 1;
        }
    }

    while out.len() < 4 {
        out.push(0);
    }

    let mut out_arr = [0u8; 4];
    out_arr.copy_from_slice(&out[..4]);
    let moved = out_arr != line;
    (out_arr, reward, moved)
}

fn get_line(board: &[u8; 16], action: u8, index: usize) -> [u8; 4] {
    match action {
        0 => [board[index], board[4 + index], board[8 + index], board[12 + index]],
        1 => [
            board[4 * index + 3],
            board[4 * index + 2],
            board[4 * index + 1],
            board[4 * index],
        ],
        2 => [board[12 + index], board[8 + index], board[4 + index], board[index]],
        _ => [board[4 * index], board[4 * index + 1], board[4 * index + 2], board[4 * index + 3]],
    }
}

fn set_line(board: &mut [u8; 16], action: u8, index: usize, line: [u8; 4]) {
    match action {
        0 => {
            board[index] = line[0];
            board[4 + index] = line[1];
            board[8 + index] = line[2];
            board[12 + index] = line[3];
        }
        1 => {
            board[4 * index + 3] = line[0];
            board[4 * index + 2] = line[1];
            board[4 * index + 1] = line[2];
            board[4 * index] = line[3];
        }
        2 => {
            board[12 + index] = line[0];
            board[8 + index] = line[1];
            board[4 + index] = line[2];
            board[index] = line[3];
        }
        _ => {
            board[4 * index] = line[0];
            board[4 * index + 1] = line[1];
            board[4 * index + 2] = line[2];
            board[4 * index + 3] = line[3];
        }
    }
}

fn parse_board(board: Vec<u8>) -> PyResult<[u8; 16]> {
    if board.len() != 16 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "board must have 16 elements",
        ));
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&board[..16]);
    Ok(arr)
}

#[pyfunction]
fn apply_action(py: Python<'_>, board: Vec<u8>, action: u8) -> PyResult<PyObject> {
    if action > 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "action must be 0..3",
        ));
    }

    let board_arr = parse_board(board)?;
    let mut next = board_arr;
    let mut reward: u32 = 0;
    let mut moved = false;

    for index in 0..4 {
        let line = get_line(&board_arr, action, index);
        let (new_line, line_reward, line_moved) = compress_line_left(line);
        set_line(&mut next, action, index, new_line);
        reward += line_reward;
        moved |= line_moved;
    }

    Ok(PyTuple::new_bound(
        py,
        [
            next.to_vec().into_py(py),
            reward.into_py(py),
            moved.into_py(py),
        ],
    )
    .into_py(py))
}

#[pyfunction]
fn legal_actions(board: Vec<u8>) -> PyResult<Vec<u8>> {
    let board_arr = parse_board(board)?;
    let mut legal = Vec::new();

    for action in 0..4u8 {
        let mut moved = false;
        for index in 0..4 {
            let line = get_line(&board_arr, action, index);
            let (_, _, line_moved) = compress_line_left(line);
            moved |= line_moved;
            if moved {
                break;
            }
        }
        if moved {
            legal.push(action);
        }
    }

    if legal.is_empty() {
        legal.push(0);
    }
    Ok(legal)
}

#[pyfunction]
fn empty_positions(board: Vec<u8>) -> PyResult<Vec<u8>> {
    let board_arr = parse_board(board)?;
    let mut positions = Vec::new();
    for (index, &value) in board_arr.iter().enumerate() {
        if value == 0 {
            positions.push(index as u8);
        }
    }
    Ok(positions)
}

#[pyfunction]
fn spawn_tile(board: Vec<u8>, position: u8, value_exp: u8) -> PyResult<Vec<u8>> {
    if position > 15 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "position must be 0..15",
        ));
    }
    if value_exp != 1 && value_exp != 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "value_exp must be 1 (tile 2) or 2 (tile 4)",
        ));
    }

    let mut board_arr = parse_board(board)?;
    if board_arr[position as usize] != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "spawn position must be empty",
        ));
    }
    board_arr[position as usize] = value_exp;
    Ok(board_arr.to_vec())
}

#[pymodule]
fn fast2048_sim(_py: Python, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(apply_action, module)?)?;
    module.add_function(wrap_pyfunction!(legal_actions, module)?)?;
    module.add_function(wrap_pyfunction!(empty_positions, module)?)?;
    module.add_function(wrap_pyfunction!(spawn_tile, module)?)?;
    Ok(())
}