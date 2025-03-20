from typing import Any, List, Type, TypeVar

T = TypeVar('T')

def assert_list_element_type(list: List[Any], element_type: Type[T]) -> List[T]:
    assert all(isinstance(element, element_type) for element in list), "All elements must be of type " + element_type.__name__
    return list